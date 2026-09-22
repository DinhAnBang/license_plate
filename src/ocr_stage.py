"""OCR candidate orchestration over retained plate crops."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .microcharnet_ocr import MicroCharNetOCR
from .plate_buffer import BufferedPlateCandidate, PlateBufferManager

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class OCRPlateCandidate:
    """V4 retained candidate plus one independent V5 OCR result."""

    track_id: int
    frame_index: int
    rank: int
    plate_class_id: int
    plate_class_name: str
    plate_confidence: float
    quality_score: float
    bbox: tuple[int, int, int, int]
    raw_text: str
    ocr_confidence: float
    char_confidences: tuple[float, ...] | None
    status: str = "ok"
    error: str | None = None


def _candidate_failure(
    candidate: BufferedPlateCandidate, rank: int, status: str, error: str
) -> OCRPlateCandidate:
    return OCRPlateCandidate(
        track_id=candidate.track_id,
        frame_index=candidate.frame_index,
        rank=rank,
        plate_class_id=candidate.plate_class_id,
        plate_class_name=candidate.plate_class_name,
        plate_confidence=float(candidate.plate_confidence),
        quality_score=float(candidate.quality.total_score),
        bbox=tuple(int(value) for value in candidate.bbox),
        raw_text="",
        ocr_confidence=0.0,
        char_confidences=None,
        status=status,
        error=error,
    )


def _ocr_one_candidate(
    ocr: MicroCharNetOCR,
    candidate: BufferedPlateCandidate,
    rank: int,
    logger: logging.Logger,
    debug: bool = False,
) -> OCRPlateCandidate:
    try:
        result = ocr.recognize(candidate.crop)
    except (TypeError, ValueError) as exc:
        logger.error(
            "OCR candidate failed: track=%s rank=%s frame=%s: %s",
            candidate.track_id,
            rank,
            candidate.frame_index,
            exc,
        )
        if hasattr(ocr, "_counters"):
            ocr._counters.failure_count += 1  # type: ignore[attr-defined]
        return _candidate_failure(candidate, rank, "decode_failed", str(exc))
    except Exception as exc:  # noqa: BLE001 - isolate one bad crop from the track/video
        if debug:
            logger.exception(
                "OCR inference failed: track=%s rank=%s frame=%s",
                candidate.track_id,
                rank,
                candidate.frame_index,
            )
        else:
            logger.error(
                "OCR inference failed: track=%s rank=%s frame=%s: %s",
                candidate.track_id,
                rank,
                candidate.frame_index,
                exc,
            )
        if hasattr(ocr, "_counters"):
            ocr._counters.failure_count += 1  # type: ignore[attr-defined]
        return _candidate_failure(candidate, rank, "decode_failed", str(exc))

    return OCRPlateCandidate(
        track_id=candidate.track_id,
        frame_index=candidate.frame_index,
        rank=rank,
        plate_class_id=candidate.plate_class_id,
        plate_class_name=candidate.plate_class_name,
        plate_confidence=float(candidate.plate_confidence),
        quality_score=float(candidate.quality.total_score),
        bbox=tuple(int(value) for value in candidate.bbox),
        raw_text=result.text,
        ocr_confidence=float(result.confidence),
        char_confidences=result.char_confidences,
        status=result.status,
        error=None,
    )


def run_ocr_on_retained(
    retained_by_track: Mapping[int, Sequence[BufferedPlateCandidate]],
    ocr: MicroCharNetOCR,
    logger: logging.Logger | None = None,
    debug: bool = False,
) -> tuple[OCRPlateCandidate, ...]:
    """OCR only the already-retained V4 candidates supplied by the caller."""

    logger = logger or LOGGER
    results: list[OCRPlateCandidate] = []
    for track_id in sorted(retained_by_track):
        candidates = retained_by_track[track_id]
        for rank, candidate in enumerate(candidates, 1):
            if int(candidate.track_id) != int(track_id):
                raise ValueError(
                    "retained_by_track key does not match candidate.track_id"
                )
            result = _ocr_one_candidate(ocr, candidate, rank, logger, debug=debug)
            if debug:
                timing = getattr(ocr, "last_timing", None)
                timing_text = (
                    f" preprocess_ms={timing.preprocess_ms:.3f}"
                    f" inference_ms={timing.inference_ms:.3f}"
                    f" decode_ms={timing.decode_ms:.3f}"
                    if timing is not None
                    else ""
                )
                print(
                    f"track={result.track_id} rank={result.rank} "
                    f"frame={result.frame_index} "
                    f"input_crop={candidate.crop.shape[1]}x{candidate.crop.shape[0]} "
                    f"preprocessed_shape={getattr(ocr, 'input_shape', '<unknown>')} "
                    f"model_output={getattr(ocr, 'output_name', '<unknown>')}"
                    f" shape={getattr(ocr, 'output_shape', '<unknown>')} "
                    f"raw_text={result.raw_text!r} "
                    f"confidence={result.ocr_confidence:.4f}{timing_text}"
                )
            results.append(result)
    return tuple(results)


def run_ocr_on_topk(
    manager: PlateBufferManager,
    ocr: MicroCharNetOCR,
    logger: logging.Logger | None = None,
    debug: bool = False,
) -> tuple[OCRPlateCandidate, ...]:
    """Bridge V4's final Top-K buffer to V5 without touching dropped crops."""

    retained = {
        summary.track_id: manager.get_top_candidates(summary.track_id)
        for summary in manager.finalize_all()
    }
    return run_ocr_on_retained(retained, ocr, logger=logger, debug=debug)


def run_ocr_on_image_candidates(
    candidates: Sequence[BufferedPlateCandidate],
    ocr: MicroCharNetOCR,
    logger: logging.Logger | None = None,
    debug: bool = False,
) -> tuple[OCRPlateCandidate, ...]:
    """OCR resolved image plates once each, with rank 1 and no tracking state.

    ``track_id`` on these adapter candidates is only an image-detection ID
    supplied by the image caller.  This function does not vote, fuse, or
    compare detections across frames.
    """

    logger = logger or LOGGER
    results: list[OCRPlateCandidate] = []
    for candidate in candidates:
        result = _ocr_one_candidate(ocr, candidate, 1, logger, debug=debug)
        if debug:
            timing = getattr(ocr, "last_timing", None)
            timing_text = (
                f" preprocess_ms={timing.preprocess_ms:.3f}"
                f" inference_ms={timing.inference_ms:.3f}"
                f" decode_ms={timing.decode_ms:.3f}"
                if timing is not None
                else ""
            )
            print(
                f"track={result.track_id} rank=1 frame={result.frame_index} "
                f"input_crop={candidate.crop.shape[1]}x{candidate.crop.shape[0]} "
                f"preprocessed_shape={getattr(ocr, 'input_shape', '<unknown>')} "
                f"model_output={getattr(ocr, 'output_name', '<unknown>')}"
                f" shape={getattr(ocr, 'output_shape', '<unknown>')} "
                f"raw_text={result.raw_text!r} "
                f"confidence={result.ocr_confidence:.4f}{timing_text}"
            )
        results.append(result)
    return tuple(results)


__all__ = ["OCRPlateCandidate", "run_ocr_on_retained", "run_ocr_on_topk", "run_ocr_on_image_candidates"]
