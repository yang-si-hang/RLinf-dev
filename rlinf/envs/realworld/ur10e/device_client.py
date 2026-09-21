# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""ZeroMQ client for the external UR10e chunk-execution service."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1
UR_ACTION_DIM = 10


class UR10eDeviceError(RuntimeError):
    """Raised when the device service rejects or cannot complete a request."""


@dataclass(frozen=True)
class UR10eObservation:
    """One synchronized observation returned by the device service."""

    state: np.ndarray
    base_image: np.ndarray
    wrist_image: np.ndarray
    capture_timestamp: float
    robot_timestamp: float
    task_description: str


@dataclass(frozen=True)
class UR10eChunkResult:
    """Result returned after an external service consumes one action chunk."""

    observation: UR10eObservation
    chunk_id: int
    executed_steps: int
    terminated: bool
    truncated: bool
    message: str = ""


class UR10eDeviceClient:
    """Async facade over a serialized ZeroMQ REQ/REP connection."""

    def __init__(self, endpoint: str, timeout_ms: int = 120_000):
        try:
            import zmq
        except ImportError as exc:
            raise ImportError(
                "UR10eDeviceClient requires pyzmq. Install the RLinf project "
                "dependencies before using the UR10e environment."
            ) from exc

        self._zmq = zmq
        self._endpoint = endpoint
        self._timeout_ms = int(timeout_ms)
        if self._timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive.")
        self._context = zmq.Context()
        self._socket = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ur10e-rpc"
        )
        self._closed = False

    def _new_socket(self):
        socket = self._context.socket(self._zmq.REQ)
        socket.setsockopt(self._zmq.LINGER, 0)
        socket.setsockopt(self._zmq.SNDTIMEO, self._timeout_ms)
        socket.setsockopt(self._zmq.RCVTIMEO, self._timeout_ms)
        socket.connect(self._endpoint)
        return socket

    def _reset_socket(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._new_socket()

    @staticmethod
    def _request_metadata(request_type: str, chunk_id: int | None) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": uuid.uuid4().hex,
            "type": request_type,
            "chunk_id": chunk_id,
        }

    def _request(
        self, metadata: dict[str, Any], payload: bytes | None = None
    ) -> list[bytes]:
        if self._closed:
            raise UR10eDeviceError("UR10e device client is closed.")
        frames = [json.dumps(metadata).encode()]
        if payload is not None:
            frames.append(payload)
        if self._socket is None:
            self._socket = self._new_socket()
        try:
            self._socket.send_multipart(frames)
            return self._socket.recv_multipart()
        except self._zmq.ZMQError as exc:
            self._reset_socket()
            raise UR10eDeviceError(
                f"UR10e device request {metadata['type']} failed: {exc}"
            ) from exc

    async def _request_async(
        self, metadata: dict[str, Any], payload: bytes | None = None
    ) -> list[bytes]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._request, metadata, payload
        )

    @staticmethod
    def _decode_observation(
        metadata: dict[str, Any], frames: list[bytes]
    ) -> UR10eObservation:
        if len(frames) != 4:
            raise UR10eDeviceError(
                f"Observation response must contain 4 frames, got {len(frames)}."
            )
        arrays = []
        for name, frame in zip(
            ("state", "base_image", "wrist_image"), frames[1:], strict=True
        ):
            descriptor = metadata.get("observation", {}).get(name)
            if not isinstance(descriptor, dict):
                raise UR10eDeviceError(f"Missing observation descriptor for {name}.")
            try:
                dtype = np.dtype(descriptor["dtype"])
                shape = tuple(int(value) for value in descriptor["shape"])
                array = np.frombuffer(frame, dtype=dtype).reshape(shape).copy()
            except (KeyError, TypeError, ValueError) as exc:
                raise UR10eDeviceError(
                    f"Invalid {name} descriptor or payload."
                ) from exc
            arrays.append(array)
        return UR10eObservation(
            state=arrays[0],
            base_image=arrays[1],
            wrist_image=arrays[2],
            capture_timestamp=float(metadata["capture_timestamp"]),
            robot_timestamp=float(metadata["robot_timestamp"]),
            task_description=str(metadata.get("task_description", "")).strip(),
        )

    @classmethod
    def _decode_response(
        cls, frames: list[bytes], request: dict[str, Any]
    ) -> tuple[dict[str, Any], UR10eObservation]:
        if not frames:
            raise UR10eDeviceError("Device service returned an empty response.")
        try:
            metadata = json.loads(frames[0])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise UR10eDeviceError(
                "Device service returned invalid JSON metadata."
            ) from exc
        if metadata.get("protocol_version") != PROTOCOL_VERSION:
            raise UR10eDeviceError(
                f"Unsupported device protocol version: {metadata.get('protocol_version')}."
            )
        if metadata.get("request_id") != request["request_id"]:
            raise UR10eDeviceError(
                "Device response request_id does not match the in-flight request."
            )
        expected_type = f"{request['type']}_RESULT"
        if metadata.get("type") != expected_type:
            raise UR10eDeviceError(
                f"Expected response type {expected_type}, got {metadata.get('type')}."
            )
        if not metadata.get("ok", False):
            raise UR10eDeviceError(
                str(metadata.get("message", "Device request failed."))
            )
        return metadata, cls._decode_observation(metadata, frames)

    async def health(self) -> dict[str, Any]:
        """Check service reachability without changing device state."""
        request = self._request_metadata("HEALTH", None)
        frames = await self._request_async(request)
        metadata = json.loads(frames[0])
        if metadata.get("protocol_version") != PROTOCOL_VERSION:
            raise UR10eDeviceError("Health response protocol version does not match.")
        if metadata.get("request_id") != request["request_id"]:
            raise UR10eDeviceError("Health response request_id does not match.")
        if metadata.get("type") != "HEALTH_RESULT":
            raise UR10eDeviceError("Health response type does not match.")
        if not metadata.get("ok", False):
            raise UR10eDeviceError(str(metadata.get("message", "Health check failed.")))
        return metadata

    async def get_initial_observation(self) -> UR10eObservation:
        """Read an observation without issuing a robot command."""
        request = self._request_metadata("GET_INITIAL_OBSERVATION", None)
        frames = await self._request_async(request)
        _, observation = self._decode_response(frames, request)
        return observation

    async def execute_chunk(
        self, actions: np.ndarray, chunk_id: int
    ) -> UR10eChunkResult:
        """Submit one complete action chunk and await its final observation."""
        actions = np.ascontiguousarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != UR_ACTION_DIM:
            raise ValueError(f"actions must have shape [horizon, {UR_ACTION_DIM}].")
        if actions.shape[0] == 0 or not np.all(np.isfinite(actions)):
            raise ValueError("actions must be non-empty and finite.")
        request = self._request_metadata("EXECUTE_CHUNK", int(chunk_id))
        request["actions"] = {"shape": list(actions.shape), "dtype": actions.dtype.str}
        frames = await self._request_async(request, actions.tobytes())
        metadata, observation = self._decode_response(frames, request)
        response_chunk_id = int(metadata.get("chunk_id", -1))
        if response_chunk_id != chunk_id:
            raise UR10eDeviceError(
                f"Chunk ID mismatch: sent {chunk_id}, received {response_chunk_id}."
            )
        executed_steps = int(metadata.get("executed_steps", -1))
        if not 0 <= executed_steps <= actions.shape[0]:
            raise UR10eDeviceError(f"Invalid executed_steps: {executed_steps}.")
        return UR10eChunkResult(
            observation=observation,
            chunk_id=response_chunk_id,
            executed_steps=executed_steps,
            terminated=bool(metadata.get("terminated", False)),
            truncated=bool(metadata.get("truncated", False)),
            message=str(metadata.get("message", "")),
        )

    def close(self) -> None:
        """Close only local RPC resources; never send a device stop command."""
        if self._closed:
            return
        self._closed = True
        # destroy(linger=0) also interrupts a currently blocked local receive.
        # This is intentionally a local transport shutdown: it does not put a
        # STOP or DISCONNECT message on the wire.
        self._context.destroy(linger=0)
        self._socket = None
        self._executor.shutdown(wait=False, cancel_futures=True)
