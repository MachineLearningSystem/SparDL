# -*- coding: utf-8 -*-
from __future__ import print_function
import heapq
from typing import Dict, List, Tuple

import numpy as np
import time
import torch
import logging
import utils
import settings
from mpi4py import MPI
from settings import logger
import sys
import math


class MESSAGE:
    STOP = 'STOP'
    RUNNING = 'RUNNING'


mpi_float16 = MPI.BYTE.Create_contiguous(2).Commit()
MPI._typedict['e'] = mpi_float16
MPI_TYPES = {np.float32: MPI.FLOAT, np.float16: mpi_float16}

# THRESHOLD = 640 * 1024 * 1024
THRESHOLD = float('inf')


def topk_threshold(tensor: torch.Tensor, k: int):
    result = tensor.topk(k, sorted=False)
    return result.values.min(), result.indices


def time_log(t, s, rank):
    if rank == 0:
        logger.info('%s, %f', str(s), time.time() - t)
    return time.time()


# right rotate for a positive n
# left rotate for a negative n
def list_rotate(l, n):
    return l[-n:] + l[:-n]


def topk_sparse_allreduce(comm,
                          sparse_tensor,
                          storage,
                          indexes=None,
                          dtype=np.float32):
    tensor = sparse_tensor
    if indexes is None:
        k = int(tensor.size * 0.01)
        indexes, values = utils.topk(tensor, k)
    else:
        if not (type(indexes) is np.ndarray):
            indexes = indexes.cpu().numpy().astype(np.uint32)
        k = len(indexes)
        values = tensor  #[indexes]

    num_workers = comm.size
    if storage is not None and 'values_1d' in storage:
        values_1d = storage['values_1d']
        indexes_1d = storage['indexes_1d']
        result = storage['result']
    else:
        values_1d = np.zeros(k * num_workers, dtype=np.float32)
        indexes_1d = np.zeros(k * num_workers, dtype=np.uint32)
        result = np.zeros_like(tensor)
        storage['values_1d'] = values_1d
        storage['indexes_1d'] = indexes_1d
        storage['result'] = result

    if dtype != np.float32:
        values_1d = values_1d.astype(dtype)

    result.fill(0)

    if len(indexes) == 0:
        return result, None

    nnz = k
    comm.Allgather(values, values_1d[:num_workers * nnz])
    comm.Allgather(indexes, indexes_1d[:num_workers * nnz])
    return values_1d, indexes_1d, None  #result, None


def topk(tensor, k):
    indexes = np.abs(tensor).argsort()[-k:][::-1]
    return indexes, tensor[indexes]


def gtopk_sparse_allreduce(comm,
                           sparse_tensor,
                           storage=None,
                           indexes=None,
                           dtype=np.float32):
    """
    0: 0(0) <- 1(1), 2(2) <- 3(3), 4(4) <- 5(5), 6(6) <- 7(7)
    1: 0(0) <- 2(1), 4(2) <- 6(3)
    2: 0(0) <- 4(1)
    0 -> 1
    0 -> 2, 1 -> 3
    0 -> 4, 1 -> 5, 2 -> 6, 3 -> 7
    """
    num_workers = comm.size
    rank = comm.rank

    tensor = sparse_tensor
    if indexes is None:
        k = int(tensor.size * 0.001)
        indexes, values = utils.topk(tensor, k)
    else:
        if not (type(indexes) is np.ndarray):
            indexes = indexes.cpu().numpy()
        k = len(indexes)
        values = tensor
    original_indexes = indexes
    send_values = np.concatenate((indexes, values))
    send_values[0:k] = indexes.astype(np.uint32)
    send_values[k:2 * k] = values.astype(np.float32)
    if storage is not None and 'result_v2' in storage:
        recv_values = storage['result_v2']
        if recv_values.size < k * 2:
            recv_values = np.zeros_like(send_values)
            if storage:
                storage['result_v2'] = recv_values
        recv_values = recv_values[0:k * 2]
    else:
        recv_values = np.zeros_like(send_values)
        if storage:
            storage['result_v2'] = recv_values

    num_round = int(np.log2(num_workers))
    local_rank = rank
    exist_workers = num_workers
    step = 1
    participate_ranks = range(0, num_workers, step)
    for i in range(num_round):
        if rank in participate_ranks:
            local_rank = participate_ranks.index(rank)
            if local_rank % 2 == 0:
                source = participate_ranks[local_rank + 1]
                comm.Recv([recv_values, MPI.FLOAT], source=source)
                #reqr = comm.Irecv([recv_values, MPI.FLOAT], source=source)
                #reqr.Wait()
                tmp_indexes = recv_values[0:k].astype(np.int32)
                tmp_values = recv_values[k:2 * k]

                cv, c1, c2 = np.intersect1d(indexes,
                                            tmp_indexes,
                                            assume_unique=False,
                                            return_indices=True)
                values[c1] += tmp_values[c2]
                tmp_values[c2] = 0.0

                tmp_c = np.concatenate((values, tmp_values))
                tmp_topki, tmp_topkv = utils.topk(tmp_c, k)
                first_array_indexes = tmp_topki[tmp_topki < k]
                second_array_indexes = tmp_topki[tmp_topki >= k] - k
                indexes = np.concatenate((indexes[first_array_indexes],
                                          tmp_indexes[second_array_indexes]))
                values = np.concatenate((values[first_array_indexes],
                                         tmp_values[second_array_indexes]))

                send_values = np.concatenate((indexes, values))
                send_values[0:k] = indexes.astype(np.uint32)
                send_values[k:2 * k] = values.astype(np.float32)
            else:
                target = participate_ranks[local_rank - 1]
                logger.debug('[round:%d], %d(%d)->%d(%d)', i, rank, local_rank,
                             target, local_rank - 1)
                comm.Send([send_values, MPI.FLOAT], dest=target)
                #reqs = comm.Isend([send_values, MPI.FLOAT], dest=target)
                #reqs.Wait()
        exist_workers /= 2
        step *= 2
        participate_ranks = range(0, num_workers, step)
        comm.Barrier()

    if rank == 0:
        send_values = np.concatenate((indexes, values))
        indexes = indexes.astype(np.uint32)
        values = values.astype(np.float32)
        send_values[0:k] = indexes
        send_values[k:2 * k] = values
    else:
        send_values = recv_values[0:2 * k]
    comm.Bcast(send_values, root=0)
    tensor.fill(0.)
    if rank != 0:
        tmp_indexes = send_values[0:k].astype(np.uint32)
        tmp_values = send_values[k:2 * k].astype(np.float32)
        values = tmp_values
        indexes = tmp_indexes

    cv, c1, c2 = np.intersect1d(original_indexes,
                                indexes,
                                assume_unique=False,
                                return_indices=True)
    included_indexes = c1
    return values, indexes, included_indexes  # final selected values and indexes


def dense_allreduce(comm, tensor):
    result = np.zeros_like(tensor)
    op = MPI.SUM
    comm.Allreduce(tensor, result, op)
    comm.Barrier()
    return result


def _default_err_callback(new_num_workers, new_rank):
    logger.error(
        'Some process error accurs, number of workers changes to %d, my rank changes to %d',
        new_num_workers, new_rank)


def force_insert_item(d, key, val):
    if key not in d:
        d[key] = []
    d[key].append(val)


class AllReducer():

    def __init__(self,
                 named_parameters,
                 lock,
                 key_lock,
                 compression,
                 sparse=False,
                 err_callback=None,
                 layerwise_times=None,
                 sigma_scale=2.5,
                 density=0.001,
                 train_epoch=0,
                 norm_clip=None,
                 msg_queue=None,
                 msg_queue2=None,
                 writer=None):
        self._running = False
        self._msg_queue = msg_queue
        self._msg_queue2 = msg_queue2
        self._device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self._writer = writer
        self._profiling = True
        self._entries = {}
        self._keys = []
        self._outputs = {}
        self._residuals = {}
        self._sparse_storages = {}
        self._sparse_storages_topk = {}
        self._sparse = sparse
        self._sigma_scale = sigma_scale
        self._density = density
        self.train_epoch = train_epoch
        self.train_iter = 0
        # self._scale = 1.012
        self._scale = 1.092
        self._scale_global_decrease = 1.008
        self._scale_global_increase = 1.1
        self._scale_local_decrease = 1.012
        self._scale_local_increase = 1.3
        self._gaukvalue = []
        self._norm_dict = {}
        self._local_topk_dict = {}
        self._global_topk_dict = {}
        self._warmup_limit = 0
        self.communication_time = 0

        logger.info('density: %f', self._density)
        logger.info('threshold scale: %f', self._scale)
        self._comm = MPI.COMM_WORLD
        self._comm.Set_errhandler(MPI.ERRORS_RETURN)
        self._layerwise_times = layerwise_times  # L->1: Note that the layerwise time is from the last layer to the first
        _named_parameters = list(named_parameters)
        #self._named_parameters = {k: v for k, v
        #                        in _named_parameters}
        #self._default_for_reductions = {k: 1 for k, v
        #                        in _named_parameters}
        #self._sequential_keys = [k for k, v in _named_parameters]
        self._named_parameters = {
            k: v
            for k, v in _named_parameters if v.requires_grad
        }
        self._default_for_reductions = {
            k: 1
            for k, v in _named_parameters if v.requires_grad
        }
        self._sequential_keys = [
            k for k, v in _named_parameters if v.requires_grad
        ]

        self._lock = lock
        self._key_lock = key_lock
        self._compression = compression
        self._err_callback = err_callback if err_callback else _default_err_callback
        self._norm_clip = norm_clip

        self._allreduce_counter = 0
        self._local_threshold = {}
        self._global_threshold = {}
        self._boundaries = {}
        self._region_offsets = {}

        dsts = list(range(self._comm.size))
        srcs = dsts[::-1]
        dsts = list_rotate(dsts, -self._comm.rank)
        srcs = list_rotate(srcs, self._comm.rank + 1)
        self._dsts = dsts
        self._srcs = srcs

        self._generate_merged_parameters()
        self.allocate_sparse_storages()

        self._allreduce_timers = {}
        self._compression_timers = {}
        self._merge_timers = {}
        self._demerge_timers = {}
        self._h2d_times = {}
        self._d2h_times = {}
        self._profiling_norms = []

        self._allreduce_timers2 = {}
        self._compression_timers2 = {}
        self._merge_timers2 = {}
        self._demerge_timers2 = {}

        #self._dynamic_densities = [0.25, 0.16, 0.1, 0.05, 0.05, 0.05, 0.025]
        self._dynamic_densities = []  # the tuned one
        if self._dynamic_densities is not None:
            self._dynamic_densities.append(self._density)
            logger.info('dynamic densities = %s', self._dynamic_densities)
        self.reset()
        self.Group_num = 1  ##############################################################################################################
        self.num_workers = self.size() // self.Group_num
        # self._spar_local_threshold = [{} for j in range(self.num_workers)]
        self._spar_local_threshold = torch.empty(self.num_workers,
                                                 device=self._device)
        self._spar_global_threshold = [[
            {} for j in range(self.num_workers)
        ] for i in range(math.ceil(math.log2(self.num_workers)))]
        self._spar_region_offsets = np.zeros(self.num_workers)
        self.balanced_block_sizes = []
        self.balanced_offsets = []
        self.dest_list = []
        self.source_list = []
        self.send_start_list = []
        self.send_end_list = []
        self.recv_start_list = []
        self.recv_end_list = []
        self.last_list = []
        self._generate_arange_groups()
        self.init_send_info()
        self.timers = {}
        self.allgather_rindex_buffers = [
            np.zeros(self.chunck_size, dtype=np.int32)
            for i in range(self.num_workers)
        ]
        self.allgather_rvalue_buffers = [
            np.zeros(self.chunck_size, dtype=np.float32)
            for i in range(self.num_workers)
        ]
        self.allgather_rsize_buffers = [
            np.zeros(1, dtype=np.int32) for i in range(self.num_workers)
        ]

    def timing(self, s: str, begin: bool):
        if begin:
            self.timers[s] = self.timers.get(s, 0.) - time.time()
        else:
            self.timers[s] = self.timers.get(s, 0.) + time.time()

    def _generate_groups_with_threshold(self, threshold):
        sizes = [
            self._named_parameters[k].data.numel()
            for k in self._sequential_keys
        ][::-1]  # reverse order
        self._sizes = sizes
        print("total parameters: ", sum(sizes))
        sub_size = 0
        groups = []
        group = []
        key_groupidx_maps = {}
        idx = 0
        for k in self._sequential_keys[::-1]:
            numel = self._named_parameters[k].data.numel()
            sub_size += numel
            key_groupidx_maps[k] = idx
            if sub_size < threshold:
                group.append(k)
            else:
                idx += 1
                group.append(k)
                groups.append(group)
                group = []
                sub_size = 0
        if len(group) > 0:
            groups.append(group)
        return groups, key_groupidx_maps

    def init_send_info(self):
        Group_num = self.Group_num
        num_workers_ALL = self.size()
        rank_ALL = self.rank()
        num_workers = num_workers_ALL // Group_num
        group_rank = rank_ALL // num_workers
        self.Group_offset = num_workers * group_rank
        Group_offset = self.Group_offset
        rank = rank_ALL - Group_offset
        self.Group_rank = self.rank() // self.num_workers

        nRounds = math.ceil(math.log2(num_workers))
        extra_size = num_workers - 2**(nRounds - 1)
        size_list = []
        for i in range(0, nRounds - 1):
            new_size = 2**i
            size_list.append(int(new_size))
        if extra_size > 0:
            size_list.append(int(extra_size))
        size_list = size_list[::-1]

        # group_num = math.gcd(size_list[0], num_workers)  # group_num is m
        # items = int(num_workers // group_num)
        # intra_rank = rank % items  # sr_rank is w
        # group_rank = items * (rank // items)
        # inter_nRounds = math.ceil(math.log2(group_num))
        # intra_nRounds = nRounds - inter_nRounds

        # logger.info((rank_ALL, group_num, items, intra_rank, group_rank))

        dest_list = []
        source_list = []
        send_start_list = []
        send_end_list = []
        recv_start_list = []
        recv_end_list = []
        gap = num_workers - size_list[0]
        # logger.info((gap, num_workers, size_list, nRounds))
        for i in range(nRounds):
            dest_list.append((rank + gap) % num_workers)
            source_list.append((rank - gap) % num_workers)
            send_start_list.append((gap + rank) % num_workers)
            send_end_list.append((gap + rank + size_list[i] - 1) % num_workers)
            recv_start_list.append(rank)
            recv_end_list.append((rank + size_list[i] - 1) % num_workers)
            gap //= 2
        # logger.info((gap, group_num))
        # assert gap == group_num // 2

        rec_dest_list = []
        rec_source_list = []
        rec_send_start_list = []
        rec_send_end_list = []
        rec_recv_start_list = []
        rec_recv_end_list = []
        # rec_gap = group_num // 2
        # rec_pointer = group_num * intra_rank
        # inter_group_rank = rank // items
        # for i in range(inter_nRounds):
        #     rec_dest_list.append((inter_group_rank ^ rec_gap) * items + intra_rank)
        #     rec_source_list.append((inter_group_rank ^ rec_gap) * items + intra_rank)
        #     if rec_gap & inter_group_rank == 0:
        #         rec_recv_start_list.append(rec_pointer)
        #         rec_recv_end_list.append((rec_pointer + size_list[intra_nRounds + i] - 1) % num_workers)
        #         rec_send_start_list.append((rec_pointer + size_list[intra_nRounds + i]) % num_workers)
        #         rec_send_end_list.append((rec_pointer + size_list[intra_nRounds + i] * 2 - 1) % num_workers)
        #     else:
        #         rec_recv_start_list.append((rec_pointer + size_list[intra_nRounds + i]) % num_workers)
        #         rec_recv_end_list.append((rec_pointer + size_list[intra_nRounds + i] * 2 - 1) % num_workers)
        #         rec_send_start_list.append(rec_pointer)
        #         rec_send_end_list.append((rec_pointer + size_list[intra_nRounds + i] - 1) % num_workers)
        #         rec_pointer += size_list[intra_nRounds + i]
        #     rec_gap //= 2
        # assert rec_gap == 0

        dest_list += rec_dest_list
        source_list += rec_source_list
        send_start_list += rec_send_start_list
        send_end_list += rec_send_end_list
        recv_start_list += rec_recv_start_list
        recv_end_list += rec_recv_end_list

        self.dest_list = dest_list
        self.source_list = source_list
        self.send_start_list = send_start_list
        self.send_end_list = send_end_list
        self.recv_start_list = recv_start_list
        self.recv_end_list = recv_end_list
        self.last_list = [i for i in range(num_workers)]
        # logger.info((rank_ALL, dest_list, source_list, send_start_list, send_end_list, recv_start_list, recv_end_list))

        self.all_send_blocks = []
        self.all_recv_blocks = []
        self.all_dest = []
        self.all_source = []
        nRounds = math.ceil(math.log2(num_workers))
        for i in range(nRounds):
            dest = dest_list[i] + Group_offset
            source = source_list[i] + Group_offset
            send_start = send_start_list[i]
            send_end = send_end_list[i]
            recv_start = recv_start_list[i]
            recv_end = recv_end_list[i]
            if send_start > send_end:
                send_range0 = list(range(send_start, num_workers)) + list(
                    range(0, send_end + 1))
            else:
                send_range0 = list(range(send_start, send_end + 1))
            if recv_start > recv_end:
                recv_range0 = list(range(recv_start, num_workers)) + list(
                    range(0, recv_end + 1))
            else:
                recv_range0 = list(range(recv_start, recv_end + 1))
            for _ in range(len(send_range0)):
                self.all_send_blocks.append(send_range0[_])
                self.all_recv_blocks.append(recv_range0[_])
                self.all_dest.append(dest)
                self.all_source.append(source)

        # # sag
        self.sr_round = len(self.all_send_blocks)
        # self.sag_round = math.ceil(math.log2(self.Group_num))
        # if self.Group_num != 1:
        #     sag_round = self.sag_round
        #     step_size = self.Group_num
        #     left = 0
        #     sag_range = self.Group_num
        #     for step in range(sag_round):
        #         step_size //= 2
        #         is_left = ((self.Group_rank // step_size) == 0)
        #         if is_left:
        #             sag_dest = num_workers * ((self.Group_rank + step_size) % sag_range + left) + rank
        #         else:
        #             sag_dest = num_workers * ((self.Group_rank - step_size) % sag_range + left) + rank
        #             left += step_size
        #         if self.sr_round == 0:
        #             self.all_send_blocks.append(0)
        #             self.all_recv_blocks.append(0)
        #             self.all_dest.append(sag_dest)
        #             self.all_source.append(sag_dest)
        #         else:
        #             self.all_send_blocks.append(self.all_recv_blocks[-1])
        #             self.all_recv_blocks.append(self.all_recv_blocks[-1])
        #             self.all_dest.append(sag_dest)
        #             self.all_source.append(sag_dest)
        #         sag_range = step_size
        # # self._comm.Barrier()
        # # time.sleep(self.rank())
        # # logger.info((self.rank(), self.last_list[self.rank()], self.all_dest, self.all_send_blocks))

    def _generate_arange_groups(self):
        n = self.num_workers
        param_nums = {
            name: param.numel()
            for name, param in self._named_parameters.items()
        }
        tot = sum(param_nums.values())
        self.data_size = tot
        group_max_size = math.ceil(tot / n)
        # group_max_size = tot

        layer_heap = []  # (-left_num, name, pos)
        group_heap = []  # (size, g_id)
        tmp = []
        for i in param_nums.items():
            layer_heap.append((-i[1], i[0], 0))
        # heapq.heapify(layer_heap)
        for i in range(n):
            group_heap.append((0, i))
        # heapq.heapify(group_heap)
        groups = [{} for i in range(n)]
        group_size = np.zeros(n, np.int32)
        key_groupidx_maps: Dict[str, List[int]] = {}

        # self._merged_parameters[list(self._merged_parameters)[0]] = torch.empty((self.num_workers, group_max_size), device=self._device)

        def append_slice(group_id, left_num, append_num, pos, name):
            nonlocal groups, group_size, key_groupidx_maps
            groups[group_id][name] = (
                (pos, pos + append_num),
                (group_size[group_id], group_size[group_id] + append_num),
            )
            if pos:
                key_groupidx_maps[name].append(group_id)
            else:
                key_groupidx_maps[name] = [group_id]
            group_size[group_id] += append_num
            left_num -= append_num
            pos += append_num
            return left_num, pos

        while len(layer_heap) > 0:
            left_num, name, pos = layer_heap.pop(0)
            left_num = -left_num
            size, g_id = group_heap.pop(0)
            left_num, pos = append_slice(g_id, left_num,
                                         min(left_num, group_max_size - size),
                                         pos, name)
            if left_num > 0:
                layer_heap.insert(0, (-left_num, name, pos))
            if group_size[g_id] < group_max_size:
                group_heap.insert(0, (group_size[g_id], g_id))
        self._local_residual = None
        # self._local_residual = torch.zeros((self.num_workers, group_max_size), device=self._device)
        # self._global_residual = torch.zeros(self.num_workers, group_max_size, device=self._device)
        # self._group_buffers = torch.zeros(self.num_workers, group_max_size)
        self._spar_groups: List[Dict[str, Tuple[Tuple[int, int],
                                                Tuple[int, int]]]] = groups
        self._spar_key_groupidx_maps: Dict[str, List[int]] = key_groupidx_maps
        self.chunck_size = group_max_size
        self.h = int(self.chunck_size * self._density / self.Group_num)
        self.h_step = int((self.chunck_size * self._density - self.h) / 100)
        self.h_inertia = False
        self.chunck_topk_num = self.chunck_size
        self._named_numel = param_nums

        self._topk_gap = 10000
        self._topk_num: Dict[int, int] = {}
        self._topk_size: Dict[int, int] = {}
        for numel in param_nums.values():
            k = max(1, int(numel * self._density))
            self._topk_num[k] = self._topk_num.get(k, 0) + 1
            self._topk_size[k] = max(self._topk_size.get(k, 0), numel)
        #     self._topk_numel.setdefault(k, [0, 0])
        #     self._topk_numel[k][0] += 1
        #     self._topk_numel[k][1] = max(numel, self._topk_numel[k][1])
        self._topk_counter: Dict[int,
                                 int] = dict.fromkeys(self._topk_num.keys(), 0)
        self._topk_buffer: Dict[int, torch.Tensor] = {
            k: torch.zeros(
                (min(self._topk_num[k], self._topk_gap), self._topk_size[k]),
                device=self._device)
            for k in self._topk_num.keys()
        }
        self._topk_list: Dict[int, List[str]] = {
            k: []
            for k in self._topk_num.keys()
        }
        self._topk_threshold: Dict[int, torch.Tensor] = {
            k: torch.empty((min(self._topk_num[k], self._topk_gap), 1))
            for k in self._topk_num.keys()
        }
        self._region_size = group_size
        logger.info('spar_groups: %s', groups)
        logger.info('spar_key_groupidx_maps: %s', key_groupidx_maps)
        return

    def _generate_merged_parameters(self):
        self._merged_parameters = {}
        groups, key_groupidx_maps = self._generate_groups_with_threshold(
            THRESHOLD)
        logger.info('groups: %s', groups)
        logger.info('key_groupidx_maps: %s', key_groupidx_maps)
        new_keys = []
        self._merged_parameter_offsets = {}
        for g in groups:
            sub_size = 0
            offsets = []
            for k in g:
                offsets.append(sub_size)
                numel = self._named_parameters[k].data.numel()
                sub_size += numel
            new_key = ':'.join(g)
            new_keys.append(new_key)
            self._merged_parameters[new_key] = torch.zeros(
                sub_size,
                device=self._named_parameters[g[0]].device,
                dtype=self._named_parameters[g[0]].dtype,
                requires_grad=False)
            self._merged_parameter_offsets[new_key] = offsets
            self._allreduce_counter = 0
            self._local_threshold[new_key] = 0.0
            self._global_threshold[new_key] = 0.0
            self._boundaries[new_key] = self._comm.size * [0]
            self._region_offsets[new_key] = self._comm.size * [0]

        self._groups = groups
        self._key_groupidx_maps = key_groupidx_maps
        self._groups_flags = []
        for g in self._groups:
            flags = []
            for k in g:
                flags.append(0)
            self._groups_flags.append(flags)
        logger.info('offsets: ', self._merged_parameter_offsets)

    def _layer2group(self, tensor: torch.Tensor, name):
        group_buffers = torch.zeros((self.num_workers, self.chunck_size),
                                    device=self._device)
        group_idxs = self._spar_key_groupidx_maps[name]
        for g in group_idxs:
            (pl, pr), (gl, gr) = self._spar_groups[g][name]
            group_buffers[g][gl:gr] = tensor[pl:pr]
        return group_buffers

    def _group2layer(self, group_buffers: torch.Tensor):
        tensor_dict = {
            name: torch.zeros(numel, device=self._device)
            for name, numel in self._named_numel.items()
        }
        for g_id, g in enumerate(self._spar_groups):
            for name, ((pl, pr), (gl, gr)) in g.items():
                tensor_dict[name][pl:pr] = group_buffers[g_id][gl:gr]
        for name in tensor_dict.keys():
            tensor_dict[name] = tensor_dict[name].view_as(
                self._named_parameters[name])
        group_idx = self._key_groupidx_maps[name]
        self._groups_flags[group_idx] = [0] * len(
            self._groups_flags[group_idx])
        return tensor_dict

    def _push_to_buffer(self, name, tensor):
        if len(self._groups) == len(self._sequential_keys):
            return name, tensor
        group_idx = self._key_groupidx_maps[name]
        g = self._groups[group_idx]
        new_key = ':'.join(g)
        layer_idx = g.index(name)
        offset = self._merged_parameter_offsets[new_key][layer_idx]
        numel = tensor.data.numel()
        # logger.info((name, self._merged_parameters[new_key], numel, offset, tensor))
        self._merged_parameters[new_key].data[offset:offset +
                                              numel] = tensor.view(numel).data
        self._groups_flags[group_idx][layer_idx] = 1
        try:
            idx = self._groups_flags[group_idx].index(0)
        except:
            idx = -1
        if idx >= 0:
            return name, None
        return new_key, self._merged_parameters[new_key]

    def _pull_from_buffer(self, name, merged_tensor):
        if len(self._groups) == len(self._sequential_keys):
            return {name: merged_tensor}
        offsets = self._merged_parameter_offsets[name]
        g = name.split(':')
        group_idx = self._key_groupidx_maps[g[0]]
        self._groups_flags[group_idx] = [0] * len(
            self._groups_flags[group_idx])
        tensors = {}
        for i, k in enumerate(g):
            offset = offsets[i]
            original_tensor = self._named_parameters[k]
            numel = original_tensor.numel()
            tensor = torch.zeros(numel,
                                 device=original_tensor.device,
                                 dtype=original_tensor.dtype)
            tensor.data = merged_tensor.data[offset:offset + numel]
            tensors[k] = tensor.view(original_tensor.shape)
        return tensors

    def rank(self):
        return self._comm.rank

    def size(self):
        return self._comm.size

    def allocate_sparse_storages(self):
        for k, v in self._merged_parameters.items():
            self.allocate_storage(k, v)

    def _print_profiling(self):
        if self._profiling and self.rank() == 0 and len(
                self._allreduce_timers.keys()) > 0 and len(
                    self._allreduce_timers.get(
                        list(self._allreduce_timers.keys())[0], [])) == 50:
            cts = self._layerwise_times  # gpu computation
            mgs = self._merge_timers  # merge_times
            if len(self._compression_timers) != 0:
                cps = self._compression_timers  # compression
            ars = self._allreduce_timers  # allreduce times
            dms = self._demerge_timers  # demerge times
            d2hs = self._d2h_times
            h2ds = self._h2d_times
            l = 0
            logger.info(
                '[rank:%d]name[size]: backward, merge, compression, allreduce, demerge, d2h, h2d'
            )
            total_sz = 0
            total_ct = 0.0
            total_mg = 0.0
            total_cp = 0.0
            total_ar = 0.0
            total_dm = 0.0
            total_d2h = 0.0
            total_h2d = 0.0

            for g in self._groups:
                ct = 0.0
                sz = 0
                for k in g:
                    if cts is not None:
                        ct += cts[l]
                    else:
                        ct = 0.0
                    sz += self._sizes[l]
                    total_ct += ct
                    l += 1
                total_sz += sz
                k = ':'.join(g)
                mg = np.mean(mgs[k])
                total_mg += mg
                if len(self._compression_timers) != 0:
                    cp = np.mean(cps[k])
                    total_cp += cp
                ar = np.mean(ars[k])
                total_ar += ar
                dm = np.mean(dms[k])
                total_dm += dm
                d2h = np.mean(d2hs.get(k, [0.0]))
                total_d2h += d2h
                h2d = np.mean(h2ds.get(k, [0.]))
                total_h2d += h2d

                # if len(self._compression_timers) != 0:
                #     logger.info('[rank:%d]%s[%d]: %f,%f,%f,%f,%f,%f,%f', self.rank(), k[0:3]+'...'+k[-3:], sz, ct,mg,cp,ar,dm,d2h,h2d)
                # else:
                #     logger.info('[rank:%d]%s[%d]: %f,%f,%f,%f,%f,%f', self.rank(), k[0:3]+'...'+k[-3:], sz, ct,mg,ar,dm,d2h,h2d)
                mgs.pop(k, None)

                if len(self._compression_timers) != 0:
                    cps.pop(k, None)
                ars.pop(k, None)
                dms.pop(k, None)
                d2hs.pop(k, None)
                h2ds.pop(k, None)
            logger.info('[rank:%d]%s[%d]: %f,%f,%f,%f,%f,%f,%f', self.rank(),
                        'total', total_sz, total_ct, total_mg, total_cp,
                        total_ar, total_dm, total_d2h, total_h2d)

    def reset(self):
        self._for_reductions = self._default_for_reductions.copy()
        self._print_profiling()

    def add_tensor(self, name, tensor):
        if name in self._entries:
            return
        self._entries[name] = tensor
        return name

    def get_current_density(self):
        density = self._density
        if self._dynamic_densities is not None:
            if self.train_epoch >= len(self._dynamic_densities):
                density = self._dynamic_densities[-1]
            else:
                density = self._dynamic_densities[self.train_epoch]
        return density

    def get_approximate_sigma_scale(self, density):
        sigma_scale = 1
        if density > 0.7:
            sigma_scale = 0.5
        elif density <= 0.7 and density > 0.05:
            sigma_scale = 1.5
        elif density <= 0.05 and density > 0.01:
            sigma_scale = 2.0
        else:
            sigma_scale = 3.0
        return sigma_scale

    def get_result(self, name):
        return self._outputs[name]

    def allocate_storage(self, name, tensor):
        storage = {}
        self._sparse_storages[name] = storage
        self._sparse_storages_topk[name] = {}

    def _sparse_allreduce(self,
                          name,
                          tensor,
                          selected_tensor,
                          original_shape,
                          topk_indexes=None):
        stime = time.time()
        ct = selected_tensor
        if ct.is_cuda:  # only transfer the selected k values through PCI-e
            entry = ct.data.cpu().numpy()
        else:
            entry = ct.data.numpy()
        if self._profiling:
            force_insert_item(self._d2h_times, name, time.time() - stime)

        result = None
        included_indexes = None
        full_mean = None
        full_var = None

        if self._compression.name in ['topkA', 'topkA2']:
            result, global_indexes, included_indexes = topk_sparse_allreduce(
                self._comm,
                entry,
                self._sparse_storages[name],
                indexes=topk_indexes,
                dtype=np.float32)
        elif self._compression.name in ['gtopk']:
            result, global_indexes, included_indexes = gtopk_sparse_allreduce(
                self._comm,
                entry,
                storage=self._sparse_storages[name],
                indexes=topk_indexes,
                dtype=np.float32)

        r = torch.from_numpy(result)
        gi = torch.from_numpy(global_indexes.astype(np.int64))
        stime = time.time()
        if tensor.is_cuda:
            r = r.cuda(tensor.device, non_blocking=False)
            final_indexes = gi.cuda(tensor.device, non_blocking=False)
        else:
            final_indexes = gi

        tensor.fill_(0.0)
        if self._compression.name in ['gtopk']:
            tensor[final_indexes] = r
        elif self._compression.name in ['topkA', 'topkA2']:
            num_workers = self._comm.size
            nnz = topk_indexes.size(0)
            for i in range(num_workers):
                index = final_indexes[i * nnz:(i + 1) * nnz]
                tensor[index] += r[i * nnz:(i + 1) * nnz]
            if self._compression.name == 'topkA2':
                values, indexes = torch.topk(torch.abs(tensor.data), k=nnz)
                cv, c1, c2 = np.intersect1d(indexes.cpu().numpy(),
                                            topk_indexes.cpu().numpy(),
                                            assume_unique=False,
                                            return_indices=True)
                included_indexes = c2
                values = tensor.data[indexes]
                tensor.data.fill_(0.0)
                tensor.data[indexes] = values.data

        tensor /= self.size()
        if self._profiling:
            force_insert_item(self._h2d_times, name, time.time() - stime)
        return tensor, included_indexes, full_mean

    def _dense_allreduce(self, name, tensor):
        ct = tensor
        shape = tensor.shape
        if ct.is_cuda:
            entry = ct.data.cpu().numpy()
        else:
            entry = ct.data.numpy()

        result = dense_allreduce(self._comm, entry)

        result = result.reshape(shape)
        r = torch.from_numpy(result)
        if tensor.is_cuda:
            r = r.cuda(tensor.device, non_blocking=False)
        r /= self.size()
        return r

    @torch.no_grad()
    def run(self):
        # return
        self._running = True
        logger.info('Allreducer thread started ...')
        comm = self._comm
        flag = 1
        t = None
        while self._running:
            name = self._msg_queue.get()
            if name == 'STOP':
                break
            if name is not None:
                tensor = self._entries[name].view(-1)
                if flag == 1:
                    t = time.time()
                    flag = 0

                # push the tensor to the buffer
                stime = time.time()
                new_name, new_tensor = self._push_to_buffer(name, tensor)
                if self._profiling:
                    force_insert_item(self._merge_timers, new_name,
                                      time.time() - stime)

                if new_tensor is None:
                    continue
                flag = 1

                num_workers = comm.size
                rank_ALL = comm.rank
                device = self._device

                stime = time.time()
                if self._sparse and self._compression.name == 'spardl':
                    num_workers = self.num_workers
                    density = self._density
                    new_tensor: torch.Tensor
                    split_tensors: torch.Tensor = torch.cat(
                        (new_tensor,
                         torch.zeros(self.chunck_size * num_workers -
                                     new_tensor.numel(),
                                     device=device))).view(num_workers, -1)
                    k = int(self.chunck_size * density)
                    force_insert_item(self._merge_timers2, new_name,
                                      time.time() - t)
                    # t = time_log(t, 'group', rank_ALL)
                    cstime = time.time()
                    if self._local_residual is not None:
                        split_tensors += self._local_residual
                    self._local_residual = split_tensors.clone()
                    split_tensors_o = split_tensors.clone()
                    whole_value_rbuffers = np.zeros(self.chunck_size *
                                                    (num_workers),
                                                    dtype='float32')
                    whole_index_rbuffers = np.zeros(self.chunck_size *
                                                    (num_workers),
                                                    dtype='int32')
                    all_value_sbuffers = [None] * num_workers
                    all_index_sbuffers = [None] * num_workers
                    all_value_rbuffers = []
                    all_index_rbuffers = []
                    all_size_rbuffers = []
                    for i in range(self.sr_round):
                        if i < self.sr_round - 1:
                            all_index_rbuffers.append(
                                whole_index_rbuffers[self.chunck_size *
                                                     i:self.chunck_size *
                                                     (i + 1)])
                            all_value_rbuffers.append(
                                whole_value_rbuffers[self.chunck_size *
                                                     i:self.chunck_size *
                                                     (i + 1)])
                            all_size_rbuffers.append(np.array([0]))
                        else:
                            all_index_rbuffers.append(
                                whole_index_rbuffers[self.chunck_size * i:])
                            all_value_rbuffers.append(
                                whole_value_rbuffers[self.chunck_size * i:])
                            all_size_rbuffers.append(np.array([0]))

                    cstime = time.time()
                    g_id = self.all_send_blocks[0] if len(
                        self.all_send_blocks) else 0
                    if num_workers == 1:
                        _, g_indexes = torch.topk(split_tensors[g_id].abs(),
                                                  self.h,
                                                  sorted=False)
                    else:
                        _, g_indexes = torch.topk(split_tensors[g_id].abs(),
                                                  k,
                                                  sorted=False)
                    thres = _.min()
                    self._spar_local_threshold[g_id] = thres
                    send_index_buffer = g_indexes.view(-1)
                    send_value_buffer = split_tensors[g_id][g_indexes]
                    all_index_sbuffers[g_id] = send_index_buffer.cpu().numpy(
                    ).astype(np.int32)
                    all_value_sbuffers[g_id] = send_value_buffer.cpu().numpy(
                    ).astype(np.float32)
                    # self._local_residual[g_id] = split_tensors[g_id].clone()
                    self._local_residual[g_id][send_index_buffer] = 0.

                    compress_t1 = time.time() - cstime
                    compress_t2 = 0
                    last_list = self.last_list
                    reqs = []
                    last_block = 0
                    sr_time = time.time()
                    h = self.h
                    # logger.info(h)
                    # for i in range(len(self.all_send_blocks)):
                    for i in range(self.sr_round):
                        # logger.info((i, 'send', self.rank(), '->', self.all_dest[i]))
                        # logger.info((i, 'recv', self.rank(), '<-', self.all_source[i]))
                        reqs = []
                        sr_one_time = time.time()
                        reqs.append(
                            comm.Isend([
                                all_index_sbuffers[self.all_send_blocks[i]],
                                MPI.INT
                            ],
                                       dest=self.all_dest[i],
                                       tag=1))
                        reqs.append(
                            comm.Irecv([all_index_rbuffers[i], MPI.INT],
                                       source=self.all_source[i],
                                       tag=1))
                        reqs.append(
                            comm.Isend([
                                all_value_sbuffers[self.all_send_blocks[i]],
                                MPI.FLOAT
                            ],
                                       dest=self.all_dest[i],
                                       tag=2))
                        reqs.append(
                            comm.Irecv([all_value_rbuffers[i], MPI.FLOAT],
                                       source=self.all_source[i],
                                       tag=2))
                        reqs.append(
                            comm.Isend([
                                np.array([
                                    all_index_sbuffers[
                                        self.all_send_blocks[i]].size
                                ]).astype(np.int32), MPI.INT
                            ],
                                       dest=self.all_dest[i],
                                       tag=3))
                        reqs.append(
                            comm.Irecv([all_size_rbuffers[i], MPI.INT],
                                       source=self.all_source[i],
                                       tag=3))
                        # logger.info(('sr once: ', time.time()-sr_one_time, ' time: ', i))
                        MPI.Request.Waitall(reqs)

                        if self.Group_num == 1 or (self.Group_num > 1
                                                   and i < self.sr_round):
                            size = int(all_size_rbuffers[i])
                            rindex = all_index_rbuffers[i][:size]
                            rvalue = all_value_rbuffers[i][:size]
                            rindex = torch.from_numpy(rindex).cuda(
                                device, non_blocking=False).long()
                            rvalue = torch.from_numpy(rvalue).cuda(
                                device, non_blocking=False)
                            split_tensors[
                                self.all_recv_blocks[i]][rindex] += rvalue

                            ctime = time.time()
                            if i < self.sr_round - 1:
                                next_block = self.all_send_blocks[i + 1]
                                g_id = next_block
                            else:
                                last_block = self.all_recv_blocks[i]
                                g_id = last_block
                            if i < self.sr_round - 1:
                                _, g_indexes = torch.topk(
                                    split_tensors[g_id].abs(), k, sorted=False)
                            else:
                                _, g_indexes = torch.topk(
                                    split_tensors[g_id].abs(), h, sorted=False)
                            thres = _.min()
                            self._spar_local_threshold[g_id] = thres
                            send_index_buffer = g_indexes.view(-1)
                            send_value_buffer = split_tensors[g_id][g_indexes]
                            all_index_sbuffers[g_id] = send_index_buffer.cpu(
                            ).numpy().astype(np.int32)
                            all_value_sbuffers[g_id] = send_value_buffer.cpu(
                            ).numpy().astype(np.float32)
                            self._local_residual[g_id] = split_tensors[
                                g_id].clone()
                            self._local_residual[g_id][send_index_buffer] = 0.
                            compress_t2 += time.time() - ctime
                        # else:
                        #     size = int(all_size_rbuffers[i])
                        #     rindex = all_index_rbuffers[i][:size]
                        #     rvalue = all_value_rbuffers[i][:size]
                        #     rindex = torch.from_numpy(rindex).cuda(device, non_blocking=False).long()
                        #     rvalue = torch.from_numpy(rvalue).cuda(device, non_blocking=False)
                        #     last_block = self.all_recv_blocks[i]
                        #     # rvalue += self._local_residual[last_block][rindex]
                        #     # self._local_residual[last_block][rindex] = 0.
                        #     # logger.info((rindex.size(), rvalue.size(), send_index_buffer.size(), send_value_buffer.size()))
                        #     # logger.info((rindex.unsqueeze_(0).size(), rvalue.size(), send_index_buffer.unsqueeze_(0).size(), send_value_buffer.size()))
                        #     r_coo = torch.sparse_coo_tensor(rindex.unsqueeze_(0), rvalue, size=[self.chunck_size], dtype=torch.float32, device=device).coalesce()
                        #     s_coo = torch.sparse_coo_tensor(send_index_buffer.unsqueeze_(0), send_value_buffer, size=[self.chunck_size], dtype=torch.float32, device=device).coalesce()
                        #     # logger.info((self.rank(), 3))
                        #     s_coo = (r_coo + s_coo).coalesce()
                        #     si = s_coo.indices().squeeze_(0)
                        #     sv = s_coo.values()
                        #     if len(si) > k:
                        #         _, g_indexes = torch.topk(sv.abs(), k, sorted=False)
                        #         g_indexes = g_indexes.view(-1)
                        #         index = si[g_indexes]
                        #         send_index_buffer = index
                        #         send_value_buffer = sv[g_indexes]
                        #         all_index_sbuffers[last_block] = send_index_buffer.cpu().numpy().astype(np.int32)
                        #         all_value_sbuffers[last_block] = send_value_buffer.cpu().numpy().astype(np.float32)
                        #         sv[g_indexes] = 0.
                        #         self._local_residual[last_block][si] += (sv / 2)
                    # ------------------sag------------------
                    group_rank = rank_ALL // num_workers
                    rank = rank_ALL % num_workers
                    # sag_block = rank * self.Group_num
                    nRounds = math.ceil(math.log2(self.Group_num))
                    sag_index_buffers = [
                        np.zeros(h, np.int32) for i in range(self.Group_num)
                    ]
                    sag_value_buffers = [
                        np.zeros(h, np.float32) for i in range(self.Group_num)
                    ]
                    assert last_list[rank] == last_block
                    # _, idx = torch.topk(send_value_buffer.abs(), h, sorted=False)
                    # si = send_index_buffer[idx]
                    # sv = send_value_buffer[idx]
                    # send_value_buffer[idx] = 0.
                    # self._local_residual[last_block][send_index_buffer] += send_value_buffer
                    # send_index_buffer = si
                    # send_value_buffer = sv
                    # logger.info(send_index_buffer.numel())
                    sag_index_buffers[0] = send_index_buffer.cpu().numpy(
                    ).astype(np.int32)
                    sag_value_buffers[0] = send_value_buffer.cpu().numpy(
                    ).astype(np.float32)
                    nRounds = math.ceil(math.log2(self.Group_num))
                    for step in range(nRounds):
                        reqs = []
                        gap = 2**step
                        last_block = min(gap * 2, self.Group_num)
                        dest = (group_rank -
                                gap) % self.Group_num * num_workers + rank
                        source = (group_rank +
                                  gap) % self.Group_num * num_workers + rank
                        for b in range(last_block - gap):
                            send_block = b
                            recv_block = b + gap
                            size = h
                            # logger.info((b, last_block, recv_block))
                            reqs.append(
                                comm.Isend(
                                    sag_index_buffers[send_block][:size],
                                    dest=dest,
                                    tag=b * 2 + 0))
                            reqs.append(
                                comm.Isend(
                                    sag_value_buffers[send_block][:size],
                                    dest=dest,
                                    tag=b * 2 + 1))
                            reqs.append(
                                comm.Irecv(sag_index_buffers[recv_block],
                                           source=source,
                                           tag=b * 2 + 0))
                            reqs.append(
                                comm.Irecv(sag_value_buffers[recv_block],
                                           source=source,
                                           tag=b * 2 + 1))
                        MPI.Request.Waitall(reqs)
                    send_index_buffer = torch.from_numpy(
                        np.concatenate(sag_index_buffers)).cuda(device).long()
                    send_value_buffer = torch.from_numpy(
                        np.concatenate(sag_value_buffers)).cuda(device)
                    # logger.info(send_index_buffer.numel())
                    sag_coo = torch.sparse_coo_tensor(
                        send_index_buffer.unsqueeze_(0),
                        send_value_buffer,
                        size=[self.chunck_size],
                        dtype=torch.float32,
                        device=device).coalesce()

                    send_index_buffer = sag_coo.indices().squeeze_(0)
                    # logger.info(send_index_buffer.numel())
                    send_value_buffer = sag_coo.values()
                    if (len(send_index_buffer) > k) ^ (self.h_step > 0):
                        if self.h_inertia:
                            self.h_step *= 2
                            self.h_inertia = False
                        else:
                            self.h_inertia = True
                    else:
                        self.h_step = int(-0.5 * self.h_step) if abs(
                            self.h_step) > 1 else -self.h_step
                        self.h_inertia = False
                    self.h = max(self.h + self.h_step, int(k / self.Group_num))
                    logger.info((self.h, self.h_step, self.h_inertia,
                                 len(send_index_buffer), k))
                    logger.info(('grad size:', len(send_index_buffer), k))
                    if len(send_index_buffer) > k:
                        _, idx = torch.topk(send_value_buffer.abs(),
                                            k,
                                            sorted=False)
                        si = send_index_buffer[idx]
                        sv = send_value_buffer[idx]
                        send_value_buffer[idx] = 0.
                        self._local_residual[rank][
                            send_index_buffer] += send_value_buffer / self.Group_num
                        send_index_buffer = si
                        send_value_buffer = sv
                    send_index_buffer = send_index_buffer.cpu().numpy().astype(
                        np.int32)
                    send_value_buffer = send_value_buffer.cpu().numpy().astype(
                        np.float32)
                    force_insert_item(self._compression_timers2, new_name,
                                      compress_t1 + compress_t2)
                    force_insert_item(self._compression_timers, new_name,
                                      compress_t1 + compress_t2)
                    # if self.rank()==0:
                    #     logger.info(("sr time:", time.time()-sr_time))

                    # ——————————————————allgather——————————————————
                    ag_time = time.time()
                    rank = rank_ALL % num_workers
                    rank_bias = rank_ALL - rank
                    last_list = self.last_list
                    last_list2 = list(range(rank, num_workers))
                    if rank > 0:
                        last_list2 += list(range(0, rank))

                    self.allgather_rindex_buffers[last_list[
                        last_list2[0]]] = send_index_buffer
                    self.allgather_rvalue_buffers[last_list[
                        last_list2[0]]] = send_value_buffer
                    self.allgather_rsize_buffers[last_list[
                        last_list2[0]]][0] = send_index_buffer.size
                    nRounds = math.ceil(math.log2(num_workers))
                    for step in range(nRounds):
                        reqs = []
                        gap = 2**step
                        last_block = min(gap * 2, num_workers)
                        dest = (rank - gap) % num_workers + rank_bias
                        source = (rank + gap) % num_workers + rank_bias
                        # logger.info('rank:%d,step:%d,gap:%d,last_block:%d,dest:%d,source:%d', rank_ALL, step, gap, last_block, dest, source)
                        for b in range(last_block - gap):
                            send_block = last_list[last_list2[b]]
                            recv_block = last_list[last_list2[b + gap]]
                            size = self.allgather_rsize_buffers[send_block][0]
                            # logger.info((rank_ALL, self.allgather_rindex_buffers[send_block][:size]))
                            # logger.info((rank_ALL, b, self.allgather_rindex_buffers[send_block][:size], self.allgather_rvalue_buffers[send_block][:size], self.allgather_rsize_buffers[send_block],))
                            reqs.append(
                                comm.Isend(
                                    self.allgather_rindex_buffers[send_block]
                                    [:size],
                                    dest=dest,
                                    tag=b * 3 + 0))
                            reqs.append(
                                comm.Isend(
                                    self.allgather_rvalue_buffers[send_block]
                                    [:size],
                                    dest=dest,
                                    tag=b * 3 + 1))
                            reqs.append(
                                comm.Isend(
                                    self.allgather_rsize_buffers[send_block],
                                    dest=dest,
                                    tag=b * 3 + 2))
                            reqs.append(
                                comm.Irecv(
                                    self.allgather_rindex_buffers[recv_block],
                                    source=source,
                                    tag=b * 3 + 0))
                            reqs.append(
                                comm.Irecv(
                                    self.allgather_rvalue_buffers[recv_block],
                                    source=source,
                                    tag=b * 3 + 1))
                            reqs.append(
                                comm.Irecv(
                                    self.allgather_rsize_buffers[recv_block],
                                    source=source,
                                    tag=b * 3 + 2))
                        MPI.Request.Waitall(reqs)

                    result = torch.zeros((self.num_workers, self.chunck_size),
                                         dtype=torch.float32,
                                         device=device)
                    for i in range(num_workers):
                        # send_size = int(all_rbuffers[i][0])
                        # size = send_size // 2
                        # index = all_rbuffers[i][1:size + 1]
                        # index.dtype = np.int32
                        size = int(self.allgather_rsize_buffers[i][0])
                        if size:
                            index = self.allgather_rindex_buffers[i][:size]
                            value = self.allgather_rvalue_buffers[i][:size]
                            index = torch.from_numpy(index).to(device).long()
                            value = torch.from_numpy(value).to(device)
                            value /= self.size()
                            result[i][index] = value

                            split_tensors_o[i][index] = self._local_residual[
                                i][index]
                            self._local_residual[i] = split_tensors_o[i]
                            # self._local_residual[i][index] = 0.

                        # if self.rank()==0:
                        #     logger.info(('ag time:', time.time()-ag_time))

                self._allreduce_counter += 1
                comm_time = time.time() - stime
                self.communication_time += comm_time
                if self._profiling:
                    force_insert_item(self._allreduce_timers, new_name,
                                      time.time() - stime)
                    force_insert_item(self._allreduce_timers2, new_name,
                                      time.time() - stime)

                # Decouple on the merged gradients
                stime = time.time()
                tensors = self._pull_from_buffer(new_name, result.view(-1))
                if self._profiling:
                    force_insert_item(self._demerge_timers, new_name,
                                      time.time() - stime)
                    force_insert_item(self._demerge_timers2, new_name,
                                      time.time() - stime)
                for n in tensors:
                    self._outputs[n] = tensors[n]
                    self._entries.pop(n, None)
                    self._for_reductions.pop(n, None)
                # self._comm.Barrier()
                # time.sleep(rank_ALL)
                # logger.info(torch.sort(result))
                # return

            if len(self._for_reductions) == 0:
                self.reset()
                torch.cuda.synchronize()
                self._msg_queue2.put('DONE')

    def stop(self):
        self._running = False


def benchmark_gtopk_sparse_allreduce():
    logger.setLevel(logging.INFO)
    comm = MPI.COMM_WORLD
    rank = comm.rank
    #np.random.seed(rank)
    size = 25 * 1024 * 1024
    ratio = 0.001
    tensor = np.random.rand(size).astype(np.float32)
    k = int(tensor.size * ratio)
    indexes, values = utils.topk(tensor, k)
    #indexes, values = topk(tensor, k)
    #logger.info('topk[%d]%s', rank, values)
    tmp = tensor[indexes]
    tensor.fill(0.)
    tensor[indexes] = tmp
    logger.debug('[%d]%s', rank, tensor)
    storage = {}

    t = gtopk_sparse_allreduce(comm, tensor, storage=storage, indexes=indexes)
    iteration = 10
    stime = time.time()
    for i in range(iteration):
        t, _ = gtopk_sparse_allreduce(comm,
                                      tensor,
                                      storage=storage,
                                      indexes=indexes)
    total_time = time.time() - stime
    logger.info('average time: %f', total_time / iteration)


if __name__ == '__main__':
    benchmark_gtopk_sparse_allreduce()
