# Publishing this repository on GitHub and archiving it on Zenodo

1. **Fill in the placeholders.** Replace `AUTHOR NAME(S)`, `FAMILY-NAME`, `GIVEN-NAMES`, `AFFILIATION`, the ORCID
   and `GITHUB-USER` in `README.md`, `LICENSE`, `CITATION.cff`, `.zenodo.json` and `pyproject.toml`.
   (`grep -rn "AUTHOR\|FAMILY-NAME\|GITHUB-USER\|AFFILIATION" .` lists them.)
2. **Create the GitHub repository.** On github.com: *New repository* -> name `po-hrl` -> Public -> do not add a README.
   Then, in this folder:
   ```bash
   git init -b main
   git add .
   git commit -m "PO-HRL 1.0.0"
   git remote add origin https://github.com/GITHUB-USER/po-hrl.git
   git push -u origin main
   ```
3. **Connect Zenodo.** Sign in at <https://zenodo.org> with your GitHub account, open *Account -> GitHub*, and switch
   the `po-hrl` repository on. Zenodo reads `.zenodo.json` for the metadata.
4. **Make a release.** On GitHub: *Releases -> Draft a new release* -> tag `v1.0.0` -> title `PO-HRL 1.0.0` ->
   *Publish*. Zenodo archives the release within a few minutes and assigns a DOI.
5. **Record the DOI.** Copy the DOI from the Zenodo record into `CITATION.cff` and `README.md`, and into the
   Reproducibility section of your copy of the paper (replace `10.5281/zenodo.XXXXXXX` and the GitHub URL there), add the Zenodo DOI badge to the
   README if you like, and push. For double-blind review, keep the paper's placeholders until acceptance.
