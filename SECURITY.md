# Security policy

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability, exposed credential,
unsafe deserialization path, dependency compromise, or other report that could
help an attacker. Prefer a private GitHub security advisory for
`binggu56/pyqed` when that feature is available. Otherwise email
gubing@westlake.edu.cn with:

- the affected version or Git commit;
- operating system and Python version;
- a minimal reproduction or proof of concept;
- expected impact; and
- any suggested mitigation.

Do not include real credentials or sensitive third-party data. The project will
acknowledge and assess reports as maintainer availability permits; no response
or remediation time is guaranteed.

## Supported code

Security fixes are evaluated for the latest published release and the default
branch. Older releases may not receive patches. Scientific correctness issues
that do not create a security impact should be filed in the public issue
tracker with reproducible inputs.

## User precautions

- Install PyQED and dependencies in an isolated environment.
- Pin versions for reproducible research and review dependency updates.
- Treat pickles, NumPy object arrays, checkpoints, and input scripts from
  untrusted sources as executable or unsafe data.
- Do not commit API keys, credentials, private molecular data, or proprietary
  datasets.
- Review external executable paths before enabling optional backends.
