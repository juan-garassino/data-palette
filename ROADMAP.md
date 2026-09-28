# DataPalette Roadmap

## v1.0.0 — stable release (current)

Done for 1.0: scikit-learn removed (own estimator protocol, sklearn-compatible), label-exact
segmentation masks, seeded randomness everywhere, subcommand CLI with TOML configs, PyPI packaging
with trusted publishing, README examples executed in CI. See [CHANGELOG.md](CHANGELOG.md).

## After 1.0 (not scheduled)

- **Transforms**: ElasticDeformation, Perspective, Affine; ColorJitter, ChannelShuffle, generic CLAHE;
  RandAugment-style random composition.
- **Datasets**: VOC / COCO segmentation loaders, paired image-mask loading with format detection,
  train/val/test splitting, Hugging Face Datasets as an alternative to torchvision.
- **Performance**: vectorized batch transforms, multiprocessing in `process_directory`.
- **Docs**: API reference site; benchmarks against albumentations and torchvision.transforms.v2.

## Direction

The transform-library space is crowded (albumentations, torchvision, imgaug). DataPalette's niche is
**small, dependency-light, task-aware dataset preparation** — `GANPipeline`, `SegmentationPipeline`,
`DiffusionPipeline` and the dataset-tools prep vocabulary — rather than competing on transform count.
