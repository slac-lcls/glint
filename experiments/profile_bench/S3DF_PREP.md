# S3DF prep: mfx100848724 run 51 for the integrator benchmark

Instructions for an agent working on SLAC S3DF. The job: turn run 51's xtc2 data into files the
Perlmutter benchmark (`experiments/profile_bench`) can read without psana, collect the developers'
CrystFEL processing of the same frames, and copy everything to NERSC.

Run 51 is LCLS experimental data, not public. Before copying anything off S3DF, confirm with the
user that moving it to their NERSC scratch is allowed. Do not name the run in anything public.

## Ground rules

- Use Slurm for anything heavier than a few seconds. Do not compute on a login node.
- Do not search shared filesystems (`/sdf`, `/sdf/data`, `/sdf/group`) recursively. Find data through
  psana (`DataSource(exp=, run=)`) or paths the user or the developers give you. If you can't find
  something, stop and ask.
- Do not ask for, store or type anyone's credentials. The user does the NERSC sign-in (sshproxy) for the transfer.
- Stop and report before the transfer (step 6). Wait for the user's go-ahead.

## Why this format

The benchmark already reads SACLA "Cheetah" multi-event HDF5 (CXIDB 33 and 45) with
`stack_sacla.py` and CrystFEL's `indexamajig`. Writing run 51 in that layout means no new reader on
the NERSC side. Two constraints come from the NERSC end:

- **No lzf compression.** CrystFEL's libhdf5 there cannot read it. Use gzip (deflate) with shuffle.
- **One 2-D slab per frame, laid out exactly as the CrystFEL `.geom` describes.** Both GLINT and
  CrystFEL map pixels through that one file.

## 1. Find the inputs

Ask the user or the developers for these if you can't find them through psana. Record where each
came from.

1. **The frame set.** Use the events the developers ran CrystFEL on (their `.lst`, or the events in
   their stream), so both sides process the same frames. If they ran it on every event, use hits:
   events with at least 10 peaks from their peak finder settings. Record which choice you made.
2. **The refined CrystFEL `.geom`** for this run's Jungfrau-16M. It must be the one the developers'
   CrystFEL processing used. Do not use psana's `calibconst` geometry: on some Jungfrau-16M
   experiments it is a placeholder ("EXPERIMENT any", no per-module offsets or tilts). See
   `experiments/xtc_bridge/README.md`.
3. **The developers' CrystFEL outputs:** stream(s), the `indexamajig` command line (peak finder,
   integration radii, indexing method), the cell file, and any `partialator` / `compare_hkl` /
   `check_hkl` output.
4. **A reference model.** Ask which PDB entry or model they use for this lysozyme. Note its cell and
   space group.

## 2. Check the detector units and layout (on 5 events, in a Slurm job)

Use the psana2 environment, probably `source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh`.
Confirm the path with the user or the LCLS docs. The access pattern below is the one the repo's
reader uses (`experiments/xtc_bridge/xtc_qreader.py`):

```python
from psana import DataSource
ds = DataSource(exp="mfx100848724", run=51)
run = next(ds.runs())
det = run.Detector("jungfrau")          # confirm the detector name: run.detnames
ebeam = run.Detector("ebeamh")          # photon energy, eV: ebeam.raw.ebeamPhotonEnergy(evt)
for evt in run.events():
    calib = det.raw.calib(evt)          # expect (32, 512, 1024) float32
```

Check and record:

- **Calib's shape and units.** Histogram a few frames. The single-photon peak sits at the photon
  energy in keV if calib is in keV, or near 1 if it is already in photons. Write down which.
- **Photon energy.** Per event, `ebeamPhotonEnergy`. Reject events where it is non-finite or below
  1000 eV, which happens on some runs.
- **The slab layout.** CrystFEL geometries for the Jungfrau-16M usually describe a
  (32 × 512, 1024) = (16384, 1024) slab, module *i* at rows 512*i to 512*i + 511. Read the
  `.geom`'s `min_ss`/`max_ss`/`min_fs`/`max_fs` lines and confirm that `calib.reshape(16384, 1024)`
  puts each module where the `.geom` expects it. psana's `calib()` is in DAQ segment order, which
  need not be the `.geom`'s order. If they differ, reorder the segments, and write down the mapping.
- **The pixel mask.** Use `det.calibconst["pixel_status"]` (nonzero = bad) plus whatever mask the
  developers used for CrystFEL.

## 3. Write the frames

Write one HDF5 file per 500 exported events, named `r0051-c000.h5`, `r0051-c001.h5`, and so on. Each
file uses the Cheetah layout, one group per event:

```
/tag-<N>/data               (16384, 1024) int16, gzip level 4 + shuffle, chunked one frame
/tag-<N>/photon_energy_ev   scalar float64
```

- **Tag numbering:** `<N>` is the event's index in `run.events()`, the same numbering the repo's
  xtc readers use. Also write `/event_index` (int64, file order) and the event timestamp
  (`evt.timestamp`, uint64) as `/timestamp`, so any event can be traced back to the xtc.
- **Pixel values:** 10 ADU per photon, in integers: `round(10 * calib_in_photons)`, clipped to the
  int16 range. Set masked pixels to 0, so they're excluded by the mask file, not by their value.
  Count the clipped pixels per frame. If any frame has more than a handful, report it rather than
  silently clipping.
- **Mask:** one file, `mask.h5`, with `/data/data` (16384, 1024) uint8, where 1 = good and 0 = bad.

Run the conversion as a Slurm array, one task per output file, on a CPU partition. Use `-I` and
keep the script outside the data directories.

## 4. Write the geometry

Copy the refined `.geom` to `r0051.geom`, then change only these header lines:

```
data = /%/data
photon_energy = /%/photon_energy_ev
adu_per_photon = 10
mask_file = <basename>/mask.h5   ; or keep the developers' mask lines if they point at the same pixels
mask = /data/data
mask_good = 0x1
mask_bad = 0x0
```

Do not change `clen`, `res`, or any panel line. Record the original file's path and md5 next to it.

## 5. Validate (Slurm, on 20 events)

Show that the frames and the `.geom` agree before anything is copied:

1. **Peaks land on bright pixels.** Take 20 events that are also in the developers' stream. Their
   stream lists peaks in `fs/px ss/px` slab coordinates. In our exported frames, the mean value at
   those positions should be well above the frame median (a ratio above about 5). Report the ratio
   per event. A ratio near 1 means the layout or event mapping is wrong.
2. **CrystFEL reads the files.** Run `indexamajig` on those 20 events with `r0051.geom` and the
   developers' settings. It should find a similar number of peaks to their stream and index most of
   them. Use any CrystFEL ≥ 0.10 available on S3DF.
3. **Sizes.** Report the total size of the export. Jungfrau-16M frames are about 32 MB raw as
   int16. Sparse photon frames compress well, but measure it.

## 6. Stop and report

Before transferring, send the user:

- the frame-set choice and the number of events exported (also total events in the run, and how
  many were rejected for photon energy);
- calib's units and any segment reordering;
- the validation numbers from step 5;
- the export's total size;
- the reference model and the cell and space group;
- the list of the developers' CrystFEL files you collected.

If the export is over 2 TB, propose a subset (for example, whole chunks in event order) and wait.

## 7. Transfer to NERSC with rsync

Destination: `/pscratch/sd/k/kmdalton/mfx100848724_r51/`. Use this layout, which mirrors the
CXIDB 45 tree, so the NERSC side needs only a config file:

```
mfx100848724_r51/
  raw/r0051/          r0051-c*.h5, mask.h5
  metadata/           r0051.geom, the original .geom, MANIFEST.json, md5sums.txt
  deposit/            the developers' CrystFEL streams, .lst, cell file, command lines, merge outputs
  reference/          the reference model (.pdb or .cif)
```

1. Write `metadata/md5sums.txt` (md5 of every file) and `metadata/MANIFEST.json` with these fields:
   - experiment, run, detector name, psana version;
   - calib's units, segment order, pixel scale (10 ADU per photon), clipping counts;
   - photon energy mean and range;
   - clen and the `.geom`'s source;
   - frame-set rule, events in the run, events exported, events rejected;
   - the event-to-file map;
   - the reference model's ID, cell and space group;
   - where each deposit file came from.
2. Push from S3DF to NERSC with rsync over ssh. Run it from an S3DF data-transfer node, not a login
   or batch node; the user can confirm the hostname, probably `s3dfdtn.slac.stanford.edu`. The NERSC
   end is a NERSC data-transfer node, probably `dtn01.nersc.gov`. Confirm that `/pscratch` is mounted
   there with `ssh ... ls -d /pscratch/sd/k/kmdalton`.

   NERSC requires MFA, so the user creates the ssh key on the S3DF side with NERSC's `sshproxy`
   (`sshproxy.sh -u kmdalton` writes `~/.ssh/nersc`, valid for 24 hours). The user types the password
   and one-time code; never handle them yourself. If the copy runs past 24 hours, the user renews the
   key and you rerun the same command, which resumes.

   ```bash
   ssh -i ~/.ssh/nersc kmdalton@dtn01.nersc.gov \
       'mkdir -p /pscratch/sd/k/kmdalton/mfx100848724_r51 && chmod 700 /pscratch/sd/k/kmdalton/mfx100848724_r51'
   # in tmux or nohup, so it survives a dropped session; no -z, the frames are already gzip-compressed
   rsync -a --partial --info=progress2 --chmod=Do-rwx,Fo-rwx,Dg-rwx,Fg-rwx \
       -e "ssh -i ~/.ssh/nersc" \
       <export dir>/ kmdalton@dtn01.nersc.gov:/pscratch/sd/k/kmdalton/mfx100848724_r51/
   ```

   Rerun the same command until it transfers nothing. `md5sums.txt` covers any corruption the
   size-and-time check misses.

3. Tell the user when it finishes, with the final rsync summary. The NERSC side checks `md5sums.txt`.
   The `--chmod` above keeps the files owner-only, since the data are not public.

## What happens next on NERSC (not part of this job)

A `config_mfx.yaml` gives `raw_dir: raw/r0051`, `run_glob: 'r0051-c*'`, `adu_per_photon: 10`, the
wavelength, cell and space group. Then box, profile and CrystFEL rings run as for CXIDB 45, and the
developers' own CrystFEL result is added as a reference point.
