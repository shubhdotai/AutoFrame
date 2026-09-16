from pathlib import Path
import json
import shutil
import subprocess

import numpy as np
import pytest

from autoclip.shortform import build_reframe_plan, render_video
from autoclip.tracking import track_faces


def face(frame, x=20):
    return {'frame': frame, 'bbox': [x, 10, x + 40, 60], 'conf': 0.9}


def test_tracking_has_one_box_per_frame_and_respects_cuts():
    detections = [[face(i), face(i, 22)] for i in range(30)]
    tracks = track_faces(detections, min_track=2, scene_cuts=[15])
    assert len(tracks) == 4
    for track in tracks:
        assert len(set(track['frame'])) == len(track['frame'])
        assert not (track['frame'][0] < 15 <= track['frame'][-1])
    assert all(len(frame) == 2 for frame in detections)


def test_no_face_crop_is_exact_vertical_and_in_bounds():
    p = build_reframe_plan([], [], 50, 1920, 1080)
    assert p['crop_width'] * 16 == p['crop_height'] * 9
    assert p['crop_width'] % 2 == p['crop_height'] % 2 == 0
    assert len(set(p['crop_x1'])) == 1
    assert (p['crop_x1'] >= 0).all()
    assert (p['crop_x1'] + p['crop_width'] <= 1920).all()


def test_scene_cut_resets_held_speaker():
    t = {'frame': np.arange(15), 'bbox': np.tile([10, 10, 50, 70], (15, 1))}
    p = build_reframe_plan([t], [np.ones(15)], 30, 640, 360, scene_cuts=[15])
    assert p['source'][15] == 'center'
    assert p['track'][15] == -1
    assert p['crop_x1'][15] > p['crop_x1'][0]


def test_speaker_switch_requires_dwell():
    tracks = [{'frame': np.arange(30), 'bbox': np.tile(box, (30, 1))} for box in ([10,10,50,70], [500,10,550,70])]
    a = np.ones(30)
    b = np.zeros(30)
    b[10:13] = 2
    b[20:] = 2
    p = build_reframe_plan(tracks, [a,b], 30, 640, 360, score_smooth_window=1)
    assert np.all(p['track'][:29] == 0)
    assert p['track'][29] == 1


def test_real_checkpoint_and_detector_batch_parity():
    import torch
    from autoclip.inference import load_model, score_detector
    path = Path(__file__).resolve().parents[1] / 'models/pretrain_AVA.model'
    if not path.exists():
        pytest.skip('optional local model test; run download_models.py first')
    s = load_model(str(path), device='cpu')
    torch.manual_seed(4)
    a, v = torch.randn(1, 40, 13), torch.rand(1, 10, 112, 112) * 255
    with torch.no_grad():
        assert s.model.forward_audio_frontend(a).shape == (1,10,128)
        assert s.model.forward_visual_frontend(v).shape == (1,10,128)
    ea, ev = torch.randn(77,128), torch.randn(77,128)
    one = score_detector(s, ea, ev, duration_set=(2,4), detector_batch=1)
    many = score_detector(s, ea, ev, duration_set=(2,4), detector_batch=8)
    assert one.shape == (77,)
    assert np.isfinite(one).all()
    np.testing.assert_allclose(one, many, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='requires FFmpeg')
def test_full_video_render_retains_duration_and_audio(tmp_path):
    source = tmp_path / 'source with spaces.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=640x360:rate=25:duration=2',
                    '-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=1',
                    '-c:v','libx264','-c:a','aac',str(source)],check=True)
    p = build_reframe_plan([],[],50,640,360)
    output = tmp_path / 'vertical.mp4'
    render_video(source,source,p,output)
    result = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(output)]))
    video = next(s for s in result['streams'] if s['codec_type']=='video')
    audio = next(s for s in result['streams'] if s['codec_type']=='audio')
    assert video['codec_name']=='h264' and audio['codec_name']=='aac'
    assert video['width'] * 16 == video['height'] * 9
    assert int(video['nb_frames']) == 50
    assert abs(float(result['format']['duration']) - 2) < 0.08
    assert int(audio['sample_rate']) == 48000
