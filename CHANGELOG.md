# Changelog

All notable changes to Bun-Bench will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Additional task categories
- Improved evaluation harness
- Web-based leaderboard

---

## [0.1.0] - 2026-01-15

### Added

- **Initial release of Bun-Bench**
- 100 curated tasks covering Bun runtime issues
  - 25 Core API tasks (`Bun.serve()`, `Bun.file()`, `Bun.spawn()`, etc.)
  - 12 Fetch & Network tasks (HTTP client, WebSocket, DNS)
  - 8 SQLite tasks (`bun:sqlite` module)
  - 25 PostgreSQL driver tasks
  - 25 MySQL driver tasks
  - 10 Build & Bundle tasks (`bun build`, transpilation, HMR)
  - 8 Testing framework tasks (`bun test`)
  - 7 Package manager tasks (`bun install`, `bun pm`)

- **Task difficulty levels**
  - 30 Easy tasks
  - 45 Medium tasks
  - 25 Hard tasks

- **Task types**
  - 70 Bug Fix tasks
  - 30 Feature Implementation tasks

- **Evaluation framework**
  - Command-line interface (`bun-bench` CLI)
  - Python API for programmatic evaluation
  - Parallel evaluation support
  - Configurable timeouts and retry logic

- **Result reporting**
  - JSON, CSV, and Markdown export formats
  - Summary statistics
  - Detailed per-task results
  - Comparison reports between models

- **Documentation**
  - README with quick start guide
  - Contributing guidelines
  - API documentation
  - Evaluation guide

- **Project infrastructure**
  - MIT License
  - GitHub Actions CI/CD (planned)
  - Pre-commit hooks configuration
  - Development environment setup

### Notes

- This is the initial release for community testing
- Task set may be refined based on feedback
- Leaderboard coming soon

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 0.1.0 | 2026-01-15 | Initial release with 100 tasks |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to:
- Report bugs
- Suggest new tasks
- Submit improvements
- Join the community

---

## Links

- [GitHub Repository](https://github.com/chittihq/bun-bench)
- [Documentation](docs/index.md)
- [Issue Tracker](https://github.com/chittihq/bun-bench/issues)
