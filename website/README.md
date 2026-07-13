# PyQED homepage

The source for [pyqed.org](https://pyqed.org), the project homepage for the
open-source PyQED electronic-structure and quantum-dynamics framework. Stable
release documentation is hosted separately at
[docs.pyqed.org](https://docs.pyqed.org/en/stable/); development documentation
is available under `/en/latest/`, with the task-oriented
[User Guide](https://docs.pyqed.org/en/latest/guide/guide.html) linked directly
from the homepage.

## Local development

Requires Node.js `>=22.13.0`.

```bash
npm install
npm run dev
```

The local site runs at `http://localhost:3000`.

## Validation

```bash
npm run lint
npm test
```

`npm test` creates the production static export in `out/` and checks the
rendered pages, metadata, landmarks, and finished-site assets.

## Project shape

- `app/` contains the homepage, global styles, shared release metadata, and
  small interactive controls.
- `app/privacy/` documents that the project site performs no visitor tracking.
- `public/research/` contains project-owned scientific figures used on the site.
- `public/og-v2.png` is the current social link-preview image.
- `tests/` verifies the static production output.
- `.github/workflows/pages.yml` builds and publishes `out/` to GitHub Pages.

The production site is a static Next.js export. It has no application server,
database, project analytics, or ChatGPT Sites runtime dependency.
