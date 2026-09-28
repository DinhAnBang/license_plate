"""Read each retained video plate crop once with the production OCR engine."""

from __future__ import annotations

from ..microcharnet_ocr import MicroCharNetOCR
from ..ocr_fusion import OCRFusionCandidate
from .plate_buffer import VideoPlateBuffer


def recognize_video_topk(
    buffer: VideoPlateBuffer,
    ocr: MicroCharNetOCR,
    track_ids: tuple[int, ...],
) -> tuple[OCRFusionCandidate, ...]:
    candidates: list[OCRFusionCandidate] = []
    for track_id in track_ids:
        for rank, item in enumerate(buffer.selected(track_id), start=1):
            result = ocr.recognize(item.crop)
            candidates.append(OCRFusionCandidate(
                track_id=track_id,
                frame_index=item.candidate.frame_index,
                rank=rank,
                raw_text=result.text,
                ocr_confidence=result.confidence,
                quality_score=item.quality.total_score,
                char_confidences=result.char_confidences,
                plate_confidence=item.candidate.plate_confidence,
                status=result.status,
            ))
    return tuple(candidates)


__all__ = ["recognize_video_topk"]
