import os
import cv2
import numpy as np
from PIL import Image
import gradio as gr
import subprocess
import gc
import torch

import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# ==============================================================================
# 1. CONFIGURATION & ENVIRONMENT LAYER (환경 설정 및 종속성 검사)
# ==============================================================================
class AppConfig:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "inswapper_128.onnx")
    
    @staticmethod
    def check_ffmpeg() -> bool:
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return True
        except FileNotFoundError:
            return False

FFMPEG_OK = AppConfig.check_ffmpeg()
if not FFMPEG_OK:
    print("⚠ ffmpeg이 설치되어 있지 않거나 PATH에 등록되지 않았습니다. 오디오 기능이 제한됩니다.")


# ==============================================================================
# 2. CORE AI ENGINE LAYER (InsightFace & 이미지 처리 핵심 비즈니스 로직)
# ==============================================================================
class ColorAdjuster:
    """대상 이미지와 소스 이미지의 LAB 색 공간 기준 톤 매칭을 수행"""
    def adjust(self, swapped_face: np.ndarray, target_face: np.ndarray) -> np.ndarray:
        src_lab = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(target_face, cv2.COLOR_BGR2LAB).astype(np.float32)

        src_mean, src_std = cv2.meanStdDev(src_lab)
        tgt_mean, tgt_std = cv2.meanStdDev(tgt_lab)

        src_mean = src_mean.reshape(1, 1, 3)
        src_std = src_std.reshape(1, 1, 3)
        tgt_mean = tgt_mean.reshape(1, 1, 3)
        tgt_std = tgt_std.reshape(1, 1, 3)

        adjusted = (src_lab - src_mean) * (tgt_std / (src_std + 1e-6)) + tgt_mean
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        return cv2.cvtColor(adjusted, cv2.COLOR_LAB2BGR)


class BBoxSmoother:
    """비디오 프레임 간 바운딩 박스의 떨림 현상을 방지하기 위한 선형 보간기"""
    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self.prev = None

    def smooth(self, bbox: list) -> list:
        if self.prev is None:
            self.prev = bbox
            return bbox
        smoothed = [
            int(self.prev[i] * (1 - self.alpha) + bbox[i] * self.alpha)
            for i in range(4)
        ]
        self.prev = smoothed
        return smoothed


class FaceSwapCore:
    """InsightFace 모델 제어 및 단일 프레임 스왑/블렌딩 엔진"""
    def __init__(self):
        self.face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = get_model(AppConfig.MODEL_PATH, download=False, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.color_adjuster = ColorAdjuster()

    @staticmethod
    def feather_blend(base: np.ndarray, overlay: np.ndarray, mask: np.ndarray, feather: int = 25) -> np.ndarray:
        mask = cv2.GaussianBlur(mask, (feather, feather), 0)
        mask = mask.astype(float) / 255.0
        mask = cv2.merge([mask, mask, mask])
        return (overlay * mask + base * (1 - mask)).astype(np.uint8)

    def execute_swap(self, src_face_img: np.ndarray, frame: np.ndarray, target_face_index: int, 
                     cached_face: dict = None, frame_idx: int = 0, bbox_smoother: BBoxSmoother = None) -> tuple:
        
        src_faces = self.face_app.get(src_face_img)
        if not src_faces:
            return frame, cached_face
        src_face = src_faces[0]

        NEED_DETECT_EVERY = 5  
        if cached_face is None or frame_idx % NEED_DETECT_EVERY == 0:
            dst_faces = self.face_app.get(frame)
            if not dst_faces or target_face_index >= len(dst_faces):
                return frame, cached_face

            dst_face = dst_faces[target_face_index]
            raw_bbox = list(map(int, dst_face.bbox))
            cached_face = {"bbox": raw_bbox, "dst_face": dst_face}
        else:
            raw_bbox = cached_face["bbox"]
            dst_face = cached_face["dst_face"]

        # 비디오일 경우 bbox 스무딩 적용
        if bbox_smoother:
            x1, y1, x2, y2 = bbox_smoother.smooth(raw_bbox)
        else:
            x1, y1, x2, y2 = raw_bbox

        swapped = self.swapper.get(frame.copy(), dst_face, src_face, paste_back=True)

        # 턱 및 목 지정을 위한 가중 영역 할당(Pad) 및 크롭
        pad = int((y2 - y1) * 0.4)
        h, w = frame.shape[:2]
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)

        swapped_crop = swapped[y1p:y2p, x1p:x2p]
        target_crop = frame[y1p:y2p, x1p:x2p]

        # 색상 매칭 및 페더링 마스크 합성
        swapped_adjusted = self.color_adjuster.adjust(swapped_crop, target_crop)
        mask = np.zeros((y2p - y1p, x2p - x1p), dtype=np.uint8)
        center = ((x2p - x1p) // 2, (y2p - y1p) // 2)
        axes = (int((x2p - x1p) * 0.45), int((y2p - y1p) * 0.55))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        blended = self.feather_blend(target_crop, swapped_adjusted, mask, feather=35)

        result = frame.copy()
        result[y1p:y2p, x1p:x2p] = blended
        cached_face["bbox"] = [x1, y1, x2, y2]

        return result, cached_face

# 글로벌 엔진 인스턴스 생성
engine = FaceSwapCore()


# ==============================================================================
# 3. MEDIA PROCESSING LAYER (영상/이미지 파일 I/O, 미디어 가공 및 VRAM 관리)
# ==============================================================================
class MediaProcessor:
    @staticmethod
    def upscale_frame(frame_bgr: np.ndarray, scale: float) -> np.ndarray:
        if scale is None or scale == 1:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        return cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def merge_audio(original_video: str, swapped_video: str, output_video: str = "result_with_audio.mp4") -> str:
        cmd = [
            "ffmpeg", "-y", "-i", swapped_video, "-i", original_video,
            "-c:v", "copy", "-c:a", "copy", "-map", "0:v", "-map", "1:a?", "-shortest", output_video
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return output_video

    @staticmethod
    def process_image(src_rgb: np.ndarray, dst_rgb: np.ndarray) -> np.ndarray:
        src_bgr = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR)
        dst_bgr = cv2.cvtColor(dst_rgb, cv2.COLOR_RGB2BGR)
        
        result_bgr, _ = engine.execute_swap(src_bgr, dst_bgr, target_face_index=0)
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    @classmethod
    def process_video(cls, src_img: np.ndarray, video_path: str, face_index: int, upscale_mode: str, keep_audio: bool) -> str:
        src_face_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
        audio_active = keep_audio and FFMPEG_OK

        scale_map = {"없음": None, "1.5x": 1.5, "2x": 2.0, "4x": 4.0}
        scale = scale_map.get(upscale_mode, None)

        # ----------------------------
        # GIF Animation 처리 분기
        # ----------------------------
        if video_path.lower().endswith(".gif"):
            gif = Image.open(video_path)
            frames, durations = [], []
            cached_face = None

            for i in range(gif.n_frames):
                gif.seek(i)
                frame_bgr = cv2.cvtColor(np.array(gif.convert("RGB")), cv2.COLOR_RGB2BGR)
                swapped, cached_face = engine.execute_swap(src_face_img, frame_bgr, face_index, cached_face, frame_idx=i)
                
                if scale:
                    swapped = cls.upscale_frame(swapped, scale)
                
                frames.append(Image.fromarray(cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB)))
                durations.append(gif.info.get("duration", 40))

                if i % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            output_path = "result.gif"
            frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
            return output_path

        # ----------------------------
        # MP4 Video 처리 분기
        # ----------------------------
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_w, out_h = (int(w * scale), int(h * scale)) if scale else (w, h)

        temp_video = "result.mp4"
        out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

        cached_face = None
        bbox_smoother = BBoxSmoother(alpha=0.55)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            swapped, cached_face = engine.execute_swap(src_face_img, frame, face_index, cached_face, frame_idx, bbox_smoother)
            if scale:
                swapped = cls.upscale_frame(swapped, scale)

            out.write(swapped)
            frame_idx += 1

            # [VRAM 최적화] 무거운 인코딩 및 ONNX 텐서 축적 강제 비우기
            if frame_idx % 15 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        cap.release()
        out.release()
        torch.cuda.empty_cache()
        gc.collect()

        return cls.merge_audio(video_path, temp_video, "result_with_audio.mp4") if audio_active else temp_video


# ==============================================================================
# 4. APPLICATION / UI LAYER (Gradio 통합 인터페이스 화면 정의)
# ==============================================================================
def extract_thumbnails(video_path):
    if not video_path:
        return []
    path = video_path.name if hasattr(video_path, 'name') else video_path
    
    # 첫 프레임 추출 알고리즘
    if path.lower().endswith(".gif"):
        gif = Image.open(path)
        frame = cv2.cvtColor(np.array(gif.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        cap = cv2.VideoCapture(path)
        _, frame = cap.read()
        cap.release()

    faces = engine.face_app.get(frame)
    thumbnails = []
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        crop = cv2.resize(frame[y1:y2, x1:x2], (128, 128))
        thumbnails.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return thumbnails

css = """
.gallery-item:hover { transform: scale(1.05); border: 2px solid #4A90E2 !important; }
"""

with gr.Blocks(css=css, title="FaceSwap Studio Pro") as demo:
    gr.Markdown("# 🎭 FaceSwap Studio Pro — 통합 포트폴리오")
    gr.Markdown("개발자 역량 입증을 위해 내부 메모리 관리 아키텍처(VRAM 최적화) 및 LAB 스페이스 색보정 레이어가 고도화된 애플리케이션입니다.")

    with gr.Tabs():
        # --- TAB 1: 이미지 단일 스왑 ---
        with gr.TabItem("🖼 단일 이미지 스왑"):
            with gr.Row():
                img_src = gr.Image(label="소스 얼굴 이미지", type="numpy")
                img_dst = gr.Image(label="타겟 배경 이미지", type="numpy")
            img_btn = gr.Button("🎬 이미지 스왑 실행", variant="primary")
            img_out = gr.Image(label="결과 이미지")
            
            img_btn.click(MediaProcessor.process_image, inputs=[img_src, img_dst], outputs=[img_out])

        # --- TAB 2: 비디오 및 GIF 스왑 ---
        with gr.TabItem("🎬 비디오 / GIF 스왑"):
            with gr.Row():
                vid_src = gr.Image(label="소스 얼굴 이미지", type="numpy")
                vid_target = gr.File(label="타겟 비디오 또는 GIF")
            
            with gr.Row():
                detect_btn = gr.Button("🔍 타겟 내 얼굴 탐지")
                upscale_radio = gr.Radio(["없음", "1.5x", "2x", "4x"], value="없음", label="해상도 업스케일 엔진")
                keep_audio_chk = gr.Checkbox(label="원본 오디오 스트림 유지", value=True)

            vid_gallery = gr.Gallery(label="탐지된 얼굴 인덱스", columns=4, height=150)
            selected_face_idx = gr.Number(label="선택할 얼굴 ID (좌측부터 0번순)", value=0)
            
            vid_btn = gr.Button("🚀 비디오 스왑 및 인코딩 시작", variant="primary")
            vid_out = gr.File(label="최종 가공 비디오 결과물")

            detect_btn.click(lambda v: extract_thumbnails(v), inputs=[vid_target], outputs=[vid_gallery])
            vid_btn.click(
                lambda src, vid, idx, up, au: MediaProcessor.process_video(src, vid.name, int(idx), up, au),
                inputs=[vid_src, vid_target, selected_face_idx, upscale_radio, keep_audio_chk],
                outputs=[vid_out]
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)