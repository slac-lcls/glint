# Training the 3D-CNN peakfinder on Perlmutter

The feature re-ranker (`detector.py`) runs on CPU; the 3D U-Net (`cnn.py`) needs a
GPU. Train it on a Perlmutter A100 node. The repo imports fine without torch — only
`cnn.py` / `train_cnn.py` require it.

## 0. Get the code there
`$SCRATCH` is purged ~every 8 weeks; keep the repo in `$HOME` or re-sync. macOS
`tar`/`rsync` quirks bite — use tar-over-ssh (see the NERSC setup notes), e.g.

    tar czf - fftindex | ssh perlmutter 'cd ~/git && tar xzf -'

## 1. Environment
Use the NERSC-provided PyTorch module (CUDA-matched), no pip torch needed:

    module load pytorch
    python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

## 2. Smoke test (login node, CPU — validates the pipeline end to end)

    cd ~/git/fftindex/experiments
    python train_cnn.py --smoke        # n=32, 1 epoch, 16 samples; just checks it runs

## 3. Real training (GPU node)

    salloc -N 1 -C gpu -G 1 -q interactive -t 02:00:00 -A <account>
    module load pytorch
    cd ~/git/fftindex/experiments
    python train_cnn.py --epochs 40 --n 96 --train 4000 --batch 8 --workers 16

or as a batch job (`sbatch`) with `#SBATCH -C gpu -G 1 -t 04:00:00`. Output:
`cnn_peakfinder.pt` (`{"model": state_dict, "n": grid_size}`).

Data is generated on the fly (CPU) by the DataLoader workers; bump `--workers` so the
GPU isn't input-starved. Each sample is one simulate + 96³ FFT (~tens of ms).

## 4. Use it back in the indexer

    import torch
    from fftindex.cnn import UNet3D, CNNPeakFinder
    from fftindex import index_shot, score, simulate_shot

    ck = torch.load("cnn_peakfinder.pt", map_location="cuda")
    model = UNet3D(); model.load_state_dict(ck["model"])
    cnn = CNNPeakFinder(model, n_model=ck["n"], device="cuda")

    res = index_shot(shot.g, shot.meta["qmax"], peakfinder=cnn)   # auto-grid OK:
    #   CNNPeakFinder resamples the volume to n_model and maps peaks back via extent.

Then run `experiments/learned_vs_classical.py` / `symmetry_eval_hard.py` with `cnn`
swapped in to compare against the classical / RandomForest re-ranker.

## v1 scope & knobs
- Trained on cells ~30–64 A (fit the n=96 grid, |x| ≤ ~72 A); `CNNPeakFinder` resamples
  so larger-cell volumes still run, but accuracy there is untested — widen `cell_hi` +
  `--n`, or add multi-scale, before trusting big cells.
- Target = Gaussian blobs (`cnn_dataset.render_heatmap`) at true lattice vectors; loss
  = BCE + soft Dice for the peak/background imbalance.
- Same simulator/labels as the feature re-ranker — only the model differs.
