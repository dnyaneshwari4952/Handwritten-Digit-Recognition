# Contributing to Handwritten Digit Recognition (CNN)

Thank you for your interest in contributing to this project! We welcome contributions from everyone.

---

## 🛠️ Development Setup

1. **Fork and Clone the Repository:**
   ```bash
   git clone https://github.com/<your-username>/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Verify Installation:**
   ```bash
   pytest -v
   streamlit run streamlit_app.py
   ```

---

## 🧪 Testing Guidelines

- Run the full test suite before committing:
  ```bash
  pytest -v
  ```
- Ensure all tests pass. If you are adding a new feature or preprocessing stage, write corresponding tests in `tests/`.

---

## 🎨 Code Style

- We follow PEP 8 standards with a maximum line length of 120 characters.
- Format code using `black` and check with `ruff`:
  ```bash
  black .
  ruff check .
  ```

---

## 🚀 Submitting a Pull Request

1. Create a descriptive branch: `git checkout -b feature/my-feature` or `bugfix/issue-fix`.
2. Commit your changes with clear, meaningful commit messages.
3. Push to your fork and submit a Pull Request against `main`.
4. Follow the PR template and link any relevant issues.
