# Contributing to Travel Order Resolver

Merci de contribuer au projet Travel Order Resolver !

## Getting Started

### Setup

```bash
# Clone the repository
git clone https://github.com/Romain-Ber/TravelOrderResolver.git
cd TravelOrderResolver

# Install development dependencies
make install-dev

# Verify installation
make test
```

## Development Workflow

### Branch Naming

- `feature/` - Nouvelles fonctionnalites
- `fix/` - Corrections de bugs
- `docs/` - Documentation
- `refactor/` - Refactoring
- `test/` - Tests

Exemple: `feature/camembert-ner`, `fix/station-matching-accent`

### Commit Messages

Utiliser le format conventionnel:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: Nouvelle fonctionnalite
- `fix`: Correction de bug
- `docs`: Documentation
- `style`: Formatage (pas de changement de code)
- `refactor`: Refactoring
- `test`: Ajout de tests
- `chore`: Maintenance

Exemples:
```
feat(nlp): add CamemBERT intent classifier
fix(pathfinding): correct Dijkstra edge weight calculation
docs(readme): update installation instructions
```

### Code Style

- **Python 3.10+**
- **Black** pour le formatage (ligne max: 100)
- **isort** pour les imports
- **flake8** pour le linting
- **mypy** pour le type checking

```bash
# Format code
make format

# Run linters
make lint
```

### Testing

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-e2e
```

## Pull Request Process

1. Creer une branche depuis `main`
2. Implementer les changements
3. Ajouter/mettre a jour les tests
4. Verifier que tous les tests passent: `make test`
5. Verifier le linting: `make lint`
6. Creer une Pull Request
7. Attendre la review d'au moins un membre de l'equipe

### PR Template

```markdown
## Description
[Description des changements]

## Type of change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code formatted (`make format`)
- [ ] Linting passes (`make lint`)
```

## Project Structure

```
src/
├── nlp/          # NLP pipeline and models
├── speech/       # Speech-to-text
├── pathfinding/  # Graph algorithms
├── data/         # Data loading
├── api/          # REST API
└── utils/        # Utilities
```

## Questions?

Contacter l'equipe via les Issues GitHub ou le channel Discord du projet.
