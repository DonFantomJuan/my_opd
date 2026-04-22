# Copyright 2025 Individual Contributor: furunding
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for teacher model knowledge distillation.
"""

import time
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from recipe.gkd.teacher import TeacherClient
from verl import DataProto

teacher_topk_logps_padded, teacher_topk_indices_padded = None, None


def _cfg_get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, DictConfig):
        return OmegaConf.select(config, key, default=default)
    return config.get(key, default)


def _normalize_teacher_weights(weights, normalize_weights):
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if normalize_weights:
        total = weight_tensor.sum().item()
        if total <= 0:
            raise ValueError("Teacher weights must sum to a positive value.")
        weight_tensor = weight_tensor / total
    return weight_tensor.tolist()


class TeacherManager:
    def __init__(self, teacher_config):
        self.teacher_config = teacher_config
        self.enabled = _cfg_get(teacher_config, "enabled", True)
        self.default_n_server_workers = int(_cfg_get(teacher_config, "n_server_workers", 1))

        multi_teacher_cfg = _cfg_get(teacher_config, "multi_teacher", None)
        self.multi_teacher_enabled = bool(_cfg_get(multi_teacher_cfg, "enabled", False))
        self.mode = _cfg_get(multi_teacher_cfg, "mode", "distribution_fusion")
        self.normalize_weights = bool(_cfg_get(multi_teacher_cfg, "normalize_weights", True))
        self.truncate_merged_topk = bool(_cfg_get(multi_teacher_cfg, "truncate_merged_topk", False))
        self.merged_topk = _cfg_get(multi_teacher_cfg, "merged_topk", None)

        teacher_entries = []
        if self.multi_teacher_enabled:
            teacher_entries = _cfg_get(multi_teacher_cfg, "teachers", []) or []

        if not teacher_entries:
            teacher_entries = [
                {
                    "name": "teacher_0",
                    "server_ip": _cfg_get(teacher_config, "server_ip"),
                    "server_port": _cfg_get(teacher_config, "server_port"),
                    "weight": 1.0,
                    "n_server_workers": self.default_n_server_workers,
                }
            ]

        self.teacher_specs = []
        self.teacher_clients = []
        for idx, raw_spec in enumerate(teacher_entries):
            spec = OmegaConf.to_container(raw_spec, resolve=True) if isinstance(raw_spec, DictConfig) else dict(raw_spec)
            if not spec.get("server_ip") or spec.get("server_port") is None:
                raise ValueError(f"Teacher spec at index {idx} must provide server_ip and server_port.")

            spec.setdefault("name", f"teacher_{idx}")
            spec.setdefault("weight", 1.0)
            spec.setdefault("n_server_workers", self.default_n_server_workers)
            spec.setdefault("num_microbatches", 1)
            spec.setdefault("max_tokens", 1)
            spec.setdefault("temperature", 1)
            spec.setdefault("only_response", False)
            spec.setdefault("max_seq_len", None)

            self.teacher_specs.append(spec)
            self.teacher_clients.append(
                TeacherClient(
                    server_ip=spec["server_ip"],
                    server_port=spec["server_port"],
                    num_microbatches=spec["num_microbatches"],
                    max_tokens=spec["max_tokens"],
                    n_server_workers=spec["n_server_workers"],
                    temperature=spec["temperature"],
                    only_response=spec["only_response"],
                    max_seq_len=spec["max_seq_len"],
                )
            )

        self.weights = _normalize_teacher_weights(
            [float(spec["weight"]) for spec in self.teacher_specs], self.normalize_weights
        )
        self.names = [spec["name"] for spec in self.teacher_specs]

        if self.mode not in ("distribution_fusion", "loss_fusion"):
            raise ValueError(f"Unsupported multi_teacher.mode: {self.mode}")

    @property
    def is_multi_teacher(self):
        return len(self.teacher_clients) > 1

    @property
    def is_loss_fusion(self):
        return self.is_multi_teacher and self.mode == "loss_fusion"


def _extract_input_ids(batch: DataProto):
    input_ids = []
    attention_mask = batch.batch["attention_mask"].to(torch.bool)
    for ids, mask in zip(batch.batch["input_ids"], attention_mask, strict=False):
        input_ids.append(ids[mask].tolist())
    return input_ids, attention_mask


def _submit_teacher_requests(input_ids, teacher_client, n_server_workers):
    batch_size = len(input_ids)
    assert batch_size % n_server_workers == 0
    micro_batch_size = batch_size // n_server_workers
    futures = []
    for i in range(0, batch_size, micro_batch_size):
        futures.append(teacher_client.submit(input_ids[i : i + micro_batch_size]))
    return futures


def _collect_teacher_results(futures, teacher_name):
    all_teacher_topk_logps = []
    all_teacher_topk_indices = []
    for future in futures:
        try:
            _, teacher_topk_logps, teacher_topk_indices = future.result()
        except Exception as e:
            raise RuntimeError(f"Teacher request failed for {teacher_name}: {e}") from e
        all_teacher_topk_logps.extend(teacher_topk_logps)
        all_teacher_topk_indices.extend(teacher_topk_indices)

    return {
        "teacher_topk_logps": all_teacher_topk_logps,
        "teacher_topk_indices": all_teacher_topk_indices,
    }


def _pad_teacher_knowledge(attention_mask, teacher_topk_logps, teacher_topk_indices):
    real_seq_lens = torch.tensor([x.size(0) for x in teacher_topk_logps], dtype=torch.int32)
    topk = teacher_topk_logps[0].size(-1)

    logp_dtype = teacher_topk_logps[0].dtype
    idx_dtype = teacher_topk_indices[0].dtype
    teacher_knowledge_shape = list(attention_mask.shape) + [topk]

    global teacher_topk_logps_padded, teacher_topk_indices_padded
    if (
        teacher_topk_logps_padded is None
        or teacher_topk_logps_padded.dtype != logp_dtype
        or teacher_topk_logps_padded.shape != torch.Size(teacher_knowledge_shape)
    ):
        teacher_topk_logps_padded = torch.zeros(*teacher_knowledge_shape, dtype=logp_dtype)
    else:
        teacher_topk_logps_padded.zero_()

    if (
        teacher_topk_indices_padded is None
        or teacher_topk_indices_padded.dtype != idx_dtype
        or teacher_topk_indices_padded.shape != torch.Size(teacher_knowledge_shape)
    ):
        teacher_topk_indices_padded = torch.zeros(*teacher_knowledge_shape, dtype=idx_dtype)
    else:
        teacher_topk_indices_padded.zero_()

    batch_size = attention_mask.size(0)
    for i in range(batch_size):
        teacher_topk_logps_padded[i][attention_mask[i]] = teacher_topk_logps[i]
        teacher_topk_indices_padded[i][attention_mask[i]] = teacher_topk_indices[i]

    return {
        "real_seq_lens": real_seq_lens,
        "teacher_topk_logps": teacher_topk_logps_padded.numpy().copy(),
        "teacher_topk_indices": teacher_topk_indices_padded.numpy().copy(),
    }


def _fuse_token_distributions(token_prob_maps, target_topk):
    token_ids = []
    token_probs = []
    for token_id, prob in token_prob_maps.items():
        if prob > 0:
            token_ids.append(token_id)
            token_probs.append(prob)

    if not token_ids:
        raise RuntimeError("Merged teacher distribution is empty after fusion.")

    token_probs = np.asarray(token_probs, dtype=np.float32)
    token_probs = token_probs / token_probs.sum()

    if target_topk is not None:
        top_indices = np.argsort(-token_probs)[:target_topk]
        token_probs = token_probs[top_indices]
        token_probs = token_probs / token_probs.sum()
        token_ids = [token_ids[i] for i in top_indices]
        fused_logps = torch.full((target_topk,), torch.finfo(torch.float32).min, dtype=torch.float32)
        fused_indices = torch.zeros((target_topk,), dtype=torch.int32)
        actual_topk = len(token_ids)
        fused_logps[:actual_topk] = torch.log(torch.from_numpy(token_probs))
        fused_indices[:actual_topk] = torch.tensor(token_ids, dtype=torch.int32)
    else:
        fused_logps = torch.log(torch.from_numpy(token_probs))
        fused_indices = torch.tensor(token_ids, dtype=torch.int32)
    return fused_logps, fused_indices


def _distribution_fusion(attention_mask, teacher_outputs, weights, truncate_merged_topk, merged_topk):
    fused_teacher_topk_logps = []
    fused_teacher_topk_indices = []
    batch_size = len(teacher_outputs[0]["teacher_topk_logps"])
    if merged_topk is not None:
        target_topk = int(merged_topk)
    else:
        target_topk = max(teacher_output["teacher_topk_logps"][0].size(-1) for teacher_output in teacher_outputs)
    if truncate_merged_topk:
        target_topk = int(merged_topk) if merged_topk is not None else target_topk

    for sample_idx in range(batch_size):
        sample_logps = []
        sample_indices = []
        seq_len = teacher_outputs[0]["teacher_topk_logps"][sample_idx].size(0)
        for pos_idx in range(seq_len):
            token_prob_maps = {}
            for teacher_idx, teacher_output in enumerate(teacher_outputs):
                curr_logps = teacher_output["teacher_topk_logps"][sample_idx][pos_idx]
                curr_indices = teacher_output["teacher_topk_indices"][sample_idx][pos_idx]
                curr_probs = torch.exp(curr_logps).tolist()
                curr_indices = curr_indices.tolist()
                weight = weights[teacher_idx]
                for token_id, prob in zip(curr_indices, curr_probs, strict=False):
                    token_prob_maps[token_id] = token_prob_maps.get(token_id, 0.0) + weight * prob

            fused_logps, fused_indices = _fuse_token_distributions(
                token_prob_maps=token_prob_maps,
                target_topk=target_topk,
            )
            sample_logps.append(fused_logps.unsqueeze(0))
            sample_indices.append(fused_indices.unsqueeze(0))

        fused_teacher_topk_logps.append(torch.cat(sample_logps, dim=0))
        fused_teacher_topk_indices.append(torch.cat(sample_indices, dim=0))

    return _pad_teacher_knowledge(attention_mask, fused_teacher_topk_logps, fused_teacher_topk_indices)


def _build_single_teacher_output(attention_mask, teacher_output):
    return _pad_teacher_knowledge(
        attention_mask=attention_mask,
        teacher_topk_logps=teacher_output["teacher_topk_logps"],
        teacher_topk_indices=teacher_output["teacher_topk_indices"],
    )


def _build_loss_fusion_output(attention_mask, teacher_outputs, num_teachers, weights):
    padded_outputs = [_build_single_teacher_output(attention_mask, teacher_output) for teacher_output in teacher_outputs]
    output_batch = DataProto.from_single_dict(data={"real_seq_lens": padded_outputs[0]["real_seq_lens"]})
    output_batch.non_tensor_batch.update(
        {
            "teacher_topk_logps": padded_outputs[0]["teacher_topk_logps"],
            "teacher_topk_indices": padded_outputs[0]["teacher_topk_indices"],
            "multi_teacher_topk_logps": np.stack([item["teacher_topk_logps"] for item in padded_outputs], axis=0),
            "multi_teacher_topk_indices": np.stack([item["teacher_topk_indices"] for item in padded_outputs], axis=0),
            "multi_teacher_weights": np.asarray(weights, dtype=np.float32),
        }
    )
    output_batch.meta_info["multi_teacher"] = {"mode": "loss_fusion", "num_teachers": num_teachers}
    return output_batch


def get_teacher_knowledge(batch: DataProto, teacher_client_or_manager, n_server_workers=1, is_async=False):
    input_ids, attention_mask = _extract_input_ids(batch)
    tik1 = time.time()
    tok1 = tik1

    teacher_manager = teacher_client_or_manager if isinstance(teacher_client_or_manager, TeacherManager) else None

    if teacher_manager is None:
        futures = _submit_teacher_requests(input_ids, teacher_client_or_manager, n_server_workers)

        def cb(_future):
            nonlocal tok1
            tok1 = max(tok1, time.time())

        for future in futures:
            future.add_done_callback(cb)
    else:
        teacher_futures = []
        for spec, teacher_client in zip(teacher_manager.teacher_specs, teacher_manager.teacher_clients, strict=False):
            futures = _submit_teacher_requests(input_ids, teacher_client, int(spec["n_server_workers"]))

            def cb(_future):
                nonlocal tok1
                tok1 = max(tok1, time.time())

            for future in futures:
                future.add_done_callback(cb)
            teacher_futures.append(futures)

    def handle_futures():
        tik2 = time.time()
        if teacher_manager is None:
            teacher_output = _collect_teacher_results(futures, teacher_name="teacher_0")
            padded_output = _build_single_teacher_output(attention_mask, teacher_output)
            output_batch = DataProto.from_single_dict(data={"real_seq_lens": padded_output["real_seq_lens"]})
            output_batch.non_tensor_batch.update(
                {
                    "teacher_topk_logps": padded_output["teacher_topk_logps"],
                    "teacher_topk_indices": padded_output["teacher_topk_indices"],
                }
            )
        else:
            teacher_outputs = [
                _collect_teacher_results(futures, teacher_name=spec["name"])
                for spec, futures in zip(teacher_manager.teacher_specs, teacher_futures, strict=False)
            ]
            if not teacher_manager.is_multi_teacher:
                padded_output = _build_single_teacher_output(attention_mask, teacher_outputs[0])
                output_batch = DataProto.from_single_dict(data={"real_seq_lens": padded_output["real_seq_lens"]})
                output_batch.non_tensor_batch.update(
                    {
                        "teacher_topk_logps": padded_output["teacher_topk_logps"],
                        "teacher_topk_indices": padded_output["teacher_topk_indices"],
                    }
                )
            elif teacher_manager.is_loss_fusion:
                output_batch = _build_loss_fusion_output(
                    attention_mask=attention_mask,
                    teacher_outputs=teacher_outputs,
                    num_teachers=len(teacher_manager.names),
                    weights=teacher_manager.weights,
                )
            else:
                fused_output = _distribution_fusion(
                    attention_mask=attention_mask,
                    teacher_outputs=teacher_outputs,
                    weights=teacher_manager.weights,
                    truncate_merged_topk=teacher_manager.truncate_merged_topk,
                    merged_topk=teacher_manager.merged_topk,
                )
                output_batch = DataProto.from_single_dict(data={"real_seq_lens": fused_output["real_seq_lens"]})
                output_batch.non_tensor_batch.update(
                    {
                        "teacher_topk_logps": fused_output["teacher_topk_logps"],
                        "teacher_topk_indices": fused_output["teacher_topk_indices"],
                    }
                )
                output_batch.meta_info["multi_teacher"] = {
                    "mode": "distribution_fusion",
                    "num_teachers": len(teacher_manager.names),
                }

        tok2 = time.time()
        output_batch.meta_info["timing"] = {"get_teacher_knowledge": (tok1 - tik1) + (tok2 - tik2)}
        return output_batch

    if is_async:
        return SimpleNamespace(get=handle_futures)
    return handle_futures()
