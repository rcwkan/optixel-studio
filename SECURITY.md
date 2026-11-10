# Security Policy

## Reporting a Vulnerability

Optixel Studio takes security and user privacy seriously. If you discover a security vulnerability, please report it privately rather than opening a public issue.

Please send details of the vulnerability to:
- GitHub Security Advisories: Submit a private advisory via this repository's **Security** tab.
- Or contact the maintainers directly via repository issue requests marked confidential.

### Scope

In scope:
- Remote code execution or command injection in MCP server, CLI dispatch, or image pipeline.
- Cross-site scripting (XSS), Content Security Policy bypass, or arbitrary protocol access in the Tauri frontend.
- Credential leakages (OAuth tokens, API keys) stored in `<OPTIXEL_DATA_DIR>/credentials.json`.
- Buffer overflows, memory safety violations, or unsafe pointer dereferences in Rust image processing routines.

Out of scope:
- Denial of service via intentionally corrupted RAW files in local directories opened by the user.
- Local attacks requiring root/admin access to the user's host machine.

Thank you for helping keep Optixel Studio and its community safe!
