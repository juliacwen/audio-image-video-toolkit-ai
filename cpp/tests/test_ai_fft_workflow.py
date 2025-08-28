import importlib.util
from pathlib import Path
import numpy as np

def test_ai_fft_workflow():
    """
    End-to-end test:
    - Generates 3 WAVs
    - Converts WAV → CSV + spectrum
    - Runs NN on all spectra
    - Validates predictions
    """
    script = Path(__file__).parent / "ai_tools" / "ai_fft_workflow.py"

    # Load module and run main()
    spec = importlib.util.spec_from_file_location("ai_fft_workflow", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    preds_direct = module.main()  # returns dict {filename: array [class0,class1]}

    # Load predictions.txt
    out_dir = Path(__file__).parent / "ai_tools" / "test_output"
    pred_file = out_dir / "predictions.txt"
    assert pred_file.exists(), "predictions.txt was not created"

    # === Load from file ===
    file_names = np.loadtxt(pred_file, delimiter=",", skiprows=1, usecols=0, dtype=str)
    preds_file = np.loadtxt(pred_file, delimiter=",", skiprows=1, usecols=(1,2))

    # === Align dict predictions with file order ===
    preds_from_dict = np.array([preds_direct[f] for f in file_names])

    # === Basic assertions ===
    assert isinstance(preds_direct, dict)
    assert len(preds_direct) == 3
    for fname, prob in preds_direct.items():
        prob_arr = np.array(prob)
        assert np.all((prob_arr >= 0) & (prob_arr <= 1))

    # === Compare predictions dict vs file ===
    assert np.allclose(preds_from_dict, preds_file, atol=1e-8), "Predictions mismatch between dict and file"

    # === Multi-class softmax, check row sums ~1 ===
    preds_array = np.vstack([np.array(preds_direct[f]) for f in file_names])
    for row in preds_array:
        assert np.isclose(np.sum(row), 1.0, atol=1e-3), f"Row does not sum to 1: {row}"

    # === Ensure WAV, CSV, and spectrum files exist ===
    for suffix in ["16bit", "24bit", "float32"]:
        wav_file = out_dir / f"tone_{suffix}.wav"
        csv_file = out_dir / f"tone_{suffix}.csv"
        spectrum_file = out_dir / f"tone_{suffix}_spectrum.csv"
        assert wav_file.exists(), f"Missing WAV file: {wav_file}"
        assert csv_file.exists(), f"Missing CSV file: {csv_file}"
        assert spectrum_file.exists(), f"Missing spectrum file: {spectrum_file}"

