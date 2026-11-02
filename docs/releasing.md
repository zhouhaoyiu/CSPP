# Preparing a Public Release

1. Review the data restrictions, third-party terms and manuscript coverage with
   all authors. Confirm the software citation authors separately from manuscript
   authors. Do not add nonexistent publication or archive identifiers.
2. Run the tests and metric recalculation, and inspect the release manifest and
   privacy audit. Keep manuscripts and the cover letter outside this repository.
3. Create the intended GitHub repository under the authorized account. Review the
   file list before making it public; do not upload an entire workspace.
4. Publish a versioned archive with the identical models and a SHA-256 manifest.
   An optional Zenodo record can provide a persistent version DOI. Update links
   only once the repository/record exists and access has been tested.
5. Give editors access to the code archive. Explicitly disclose that the Chinese
   evaluation data cannot be supplied even for peer review. Do not describe the
   repository as full reproduction of those results.

The local preparation does not create a GitHub repository, register a DOI, upload
files, submit the manuscript, or bind the authors to a new data-sharing promise.

References:
- [EQTransformer repository](https://github.com/smousavi05/EQTransformer)
- [Nature data and code policy](https://www.nature.com/nature/editorial-policies/reporting-standards)
- [NC submission instructions](https://www.nature.com/ncomms/submit/how-to-submit)
