import os

from pyqed.beam import utils_drawing


def test_image_tools_receive_untrusted_paths_as_single_arguments(monkeypatch, tmp_path):
    calls = []

    def record(command, *, check):
        calls.append((command, check))

    monkeypatch.setattr(utils_drawing.subprocess, "run", record)

    image = "input; touch injected.png"
    output = "output; touch injected.png"
    utils_drawing.change_image_size(image, length="640x480", final_filename=output)
    utils_drawing.extract_image_from_video(
        "movie; touch injected.mp4",
        num_frame="[2]",
        final_filename=output,
    )

    assert calls == [
        (["convert", image, "-resize", "640x480", output], True),
        (["convert", "movie; touch injected.mp4[2]", output], True),
    ]


def test_concatenate_drawings_passes_each_file_separately(monkeypatch, tmp_path):
    calls = []
    (tmp_path / "figure one.png").touch()
    (tmp_path / "figure;two.png").touch()
    (tmp_path / "unrelated.png").touch()

    monkeypatch.setattr(
        utils_drawing.subprocess,
        "run",
        lambda command, *, check: calls.append((command, check)),
    )

    utils_drawing.concatenate_drawings(
        nx=2,
        ny=1,
        geometria_x=100,
        geometria_y=80,
        raiz="figure",
        nombreFigura="combined.png",
        directorio=tmp_path,
    )

    assert calls == [
        (
            [
                "montage",
                os.fspath(tmp_path / "figure one.png"),
                os.fspath(tmp_path / "figure;two.png"),
                "-tile",
                "2x1",
                "-geometry",
                "100x80-5-5",
                "combined.png",
            ],
            True,
        )
    ]


def test_video_encoder_receives_output_as_one_argument(monkeypatch, tmp_path):
    calls = []
    frame = tmp_path / "frame.png"
    frame.touch()
    output = "movie; touch injected.mpg"

    monkeypatch.setattr(
        utils_drawing.subprocess,
        "run",
        lambda command, *, check: calls.append((command, check)),
    )

    utils_drawing.make_video_from_file(None, [frame], filename=output)

    assert calls == [
        (
            [
                "mencoder",
                "mf://_tmp*.png",
                "-mf",
                "kind=png:fps=10",
                "-ovc",
                "lavc",
                "-lavcopts",
                "vcodec=wmv2",
                "-oac",
                "copy",
                "-o",
                output,
            ],
            True,
        )
    ]
    assert not frame.exists()
