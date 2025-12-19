# Contributing to Music AI Detector

Thank you for considering contributing to Music AI Detector! This document outlines the development workflow and standards we follow.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git

### Initial Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/Rtur2003/Music-AIDetector.git
cd Music-AIDetector
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

5. Copy and configure environment:
```bash
cp .env.example .env
```

## Development Workflow

### Branching Strategy

We follow a topic-based branching model:

- **One topic = One branch = One PR**
- Branch names format: `<category>/<description>`
- Categories: `feature`, `fix`, `docs`, `refactor`, `test`, `config`, `tooling`

Examples:
- `feature/add-genre-classification`
- `fix/memory-leak-in-processor`
- `docs/api-examples`
- `refactor/extract-validation`

### Commit Standards

Each commit must represent one atomic change:

**Format**: `<scope>: <precise technical justification>`

**Examples**:
- `config: extract hardcoded paths to Config class`
- `validation: add bounds checking for sample rate parameter`
- `refactor: isolate file I/O from feature extraction logic`
- `test: add unit tests for pitch feature extraction`

**Forbidden**:
- Mixing multiple concerns in one commit
- Vague messages like "misc", "fixes", "cleanup"
- Combining refactoring with behavior changes

### Code Quality Standards

#### Python Style

- Follow PEP 8 (enforced by flake8)
- Use Black for formatting (100 char line length)
- Sort imports with isort
- Type hints where beneficial (not mandatory for all code)

#### Running Linters

```bash
# Format code
black backend/ tests/

# Sort imports
isort backend/ tests/

# Check style
flake8 backend/ tests/

# Type check
mypy backend/

# Run all checks
pre-commit run --all-files
```

#### Code Organization

- One class per file for core components
- Separate I/O operations from business logic
- Use configuration for all paths and constants
- Add docstrings for public APIs
- Keep functions focused and single-purpose

### Testing

Tests are located in `tests/` directory.

Run tests:
```bash
python -m pytest tests/ -v
```

Test coverage:
```bash
python -m pytest tests/ --cov=backend --cov-report=html
```

### Pull Request Process

1. **Create a focused branch** for your topic
2. **Make atomic commits** following the commit standards
3. **Ensure all tests pass** and linters are satisfied
4. **Update documentation** if behavior changes
5. **Submit PR** with clear description of:
   - What changed
   - Why it changed
   - What was intentionally NOT changed
   - How to test the changes

### PR Review Checklist

Before submitting, verify:

- [ ] All commits are atomic and well-described
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No unrelated changes included
- [ ] Configuration used instead of hardcoded values
- [ ] Error handling is appropriate
- [ ] No secrets or sensitive data committed

## Architecture Guidelines

### Separation of Concerns

- **Config** (`config.py`) - All paths and constants
- **Feature Extraction** (`feature_extractor.py`) - Audio feature computation
- **Model Training** (`model_trainer.py`) - ML model training logic
- **Prediction** (`predictor.py`) - Inference pipeline
- **API** (`api.py`) - HTTP interface layer

### Adding New Features

When adding functionality:

1. Check if it belongs in an existing module
2. If new module needed, follow existing patterns
3. Use configuration for customizable values
4. Add appropriate error handling
5. Write tests for new code
6. Document public interfaces

### Error Handling

- Use appropriate exception types
- Provide helpful error messages
- Don't catch exceptions unless you can handle them
- Log errors at appropriate levels

## Questions?

Open an issue or discussion if you need clarification on:
- Architecture decisions
- Development workflow
- Testing approach
- Code organization

Thank you for contributing!
