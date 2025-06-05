import cv2 as cv
from tqdm import tqdm
from pathlib import Path
from utils.logger import setup_logger
from concurrent.futures import ThreadPoolExecutor


logger = setup_logger("__frameExtractor__")


SPLITS, FRAMES = Path("dataset/splits"), Path("dataset/processed/frames")
FRM_SKIP, MAX_WORKERS = 5, 12


def extract_frames(video: Path):
    cap = cv.VideoCapture(str(video))
    if not cap.isOpened():
        logger.error(f"Unable to open: {video}")
        return

    subpath = video.relative_to(SPLITS)
    dest = FRAMES / subpath.parent / video.stem
    dest.mkdir(parents=True, exist_ok=True)

    frames, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frames % FRM_SKIP == 0:
            vid2frames = dest / f"frame_{saved:04d}.jpg"
            cv.imwrite(str(vid2frames), frame)
            saved += 1

        frames += 1

    cap.release()


dataset = list(SPLITS.rglob("*.mp4"))
logger.info(f"Processed {len(dataset)} videos.")


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    list(
        tqdm(
            executor.map(extract_frames, dataset),
            total=len(dataset),
            desc="Processing videos",
        )
    )

logger.info("Frame extraction done.")
