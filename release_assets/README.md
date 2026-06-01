# Release Assets

The full model package is split into small parts so it can be stored in the GitHub repository without exceeding single-file limits.

Reconstruct the zip after cloning:

```bash
cat release_assets/target_drug_ml_model_package_20260601.zip.part-* > target_drug_ml_model_package_20260601.zip
sha256sum target_drug_ml_model_package_20260601.zip
unzip target_drug_ml_model_package_20260601.zip
```

Expected SHA256:

```text
59a1e5d71c3559ed1a893d52c696b76e66ba9c55a3c6451efb7a441f8ba299d7
```
