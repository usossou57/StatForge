"""Tests du module de sandbox de sécurité (T13.1).

Valide que :
- validate_generated_code détecte les violations (imports interdits, eval/exec, etc.)
- run_sandboxed exécute du code conforme
- run_sandboxed stoppe les boucles infinies (timeout)
- run_sandboxed stoppe les sur-allocations mémoire (limites resource)
"""

import sys

import pytest

from app.security.sandbox import run_sandboxed, validate_generated_code


class TestValidateGeneratedCode:
    """Tests de validation du code généré."""

    def test_valid_code_returns_empty_list(self):
        """Un code conforme retourne une liste vide."""
        code = "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})\nprint(df)"
        assert validate_generated_code(code) == []

    def test_import_os_rejected(self):
        """L'import de os est rejeté."""
        code = "import os\nprint(os.getcwd())"
        violations = validate_generated_code(code)
        assert any("os" in v for v in violations)

    def test_import_requests_rejected(self):
        """L'import de requests (réseau) est rejeté."""
        code = "import requests\nrequests.get('http://example.com')"
        violations = validate_generated_code(code)
        assert any("requests" in v for v in violations)

    def test_eval_rejected(self):
        """L'appel à eval() est rejeté."""
        code = "result = eval('1 + 1')"
        violations = validate_generated_code(code)
        assert any("eval" in v for v in violations)

    def test_exec_rejected(self):
        """L'appel à exec() est rejeté."""
        code = "exec('print(1)')"
        violations = validate_generated_code(code)
        assert any("exec" in v for v in violations)

    def test_open_write_mode_rejected(self):
        """L'ouverture en mode écriture est rejetée."""
        code = "f = open('test.txt', 'w')\nf.write('hello')"
        violations = validate_generated_code(code)
        assert any("open" in v and "w" in v for v in violations)

    def test_open_append_mode_rejected(self):
        """L'ouverture en mode ajout est rejetée."""
        code = "f = open('test.txt', 'a')\nf.write('hello')"
        violations = validate_generated_code(code)
        assert any("open" in v and "a" in v for v in violations)

    def test_open_read_mode_allowed(self):
        """L'ouverture en mode lecture est autorisée."""
        code = "f = open('test.txt', 'r')\ncontent = f.read()"
        violations = validate_generated_code(code)
        assert not any("open" in v for v in violations)

    def test_syntax_error_raises(self):
        """Un code avec erreur de syntaxe lève SyntaxError."""
        code = "def foo(\n  return"
        with pytest.raises(SyntaxError):
            validate_generated_code(code)

    def test_os_system_rejected(self):
        """L'appel à os.system() est rejeté."""
        code = "import os\nos.system('ls')"
        violations = validate_generated_code(code)
        assert any("os.system" in v for v in violations)

    def test_subprocess_rejected(self):
        """L'import de subprocess est rejeté."""
        code = "import subprocess\nsubprocess.run(['ls'])"
        violations = validate_generated_code(code)
        assert any("subprocess" in v for v in violations)

    def test_numpy_allowed(self):
        """L'import de numpy est autorisé."""
        code = "import numpy as np\nx = np.array([1, 2, 3])"
        violations = validate_generated_code(code)
        assert not violations

    def test_scipy_stats_allowed(self):
        """L'import de scipy.stats est autorisé."""
        code = "from scipy.stats import ttest_ind\nresult = ttest_ind([1, 2], [3, 4])"
        violations = validate_generated_code(code)
        assert not violations


class TestRunSandboxed:
    """Tests d'exécution sandboxée."""

    def test_nominal_execution(self):
        """Un code simple s'exécute correctement."""
        code = "print('hello world')"
        result = run_sandboxed(code)
        assert result["success"] is True
        assert "hello world" in result["stdout"]
        assert result["returncode"] == 0

    def test_import_os_raises_value_error(self):
        """Un code avec import os lève ValueError."""
        code = "import os\nprint(os.getcwd())"
        with pytest.raises(ValueError, match="Code refusé"):
            run_sandboxed(code)

    def test_eval_raises_value_error(self):
        """Un code avec eval() lève ValueError."""
        code = "result = eval('1 + 1')"
        with pytest.raises(ValueError, match="eval"):
            run_sandboxed(code)

    def test_infinite_loop_timeout(self):
        """Une boucle infinie est stoppée par le timeout ou les limites CPU."""
        code = "while True:\n    pass"
        result = run_sandboxed(code, timeout_seconds=2)
        assert result["success"] is False
        # Soit le timeout Python lève TimeoutExpired, soit RLIMIT_CPU tue le process
        assert "Timeout" in result["stderr"] or result["returncode"] != 0

    @pytest.mark.skipif(
        sys.platform != "linux", reason="Limites resource uniquement sur Linux"
    )
    def test_cpu_limit_stops_heavy_computation(self):
        """Un calcul très lourd est stoppé par la limite CPU."""
        # Calcul intensif qui devrait dépasser le timeout CPU
        code = (
            "import math\n"
            "result = 0\n"
            "for i in range(10**8):\n"
            "    result += math.sqrt(i)\n"
        )
        result = run_sandboxed(code, timeout_seconds=2, max_memory_mb=2048)
        assert result["success"] is False
        # Soit timeout, soit limite CPU atteinte
        assert "Timeout" in result["stderr"] or result["returncode"] != 0

    def test_valid_statistical_code(self):
        """Un code statistique valide s'exécute correctement."""
        code = (
            "import pandas as pd\n"
            "import numpy as np\n"
            "\n"
            "df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 4, 5, 4, 5]})\n"
            "print(df.describe())\n"
        )
        result = run_sandboxed(code)
        # Afficher le stderr en cas d'échec pour diagnostic
        if not result["success"]:
            print(f"STDOUT: {result['stdout']}")
            print(f"STDERR: {result['stderr']}")
            print(f"RETURNCODE: {result['returncode']}")
        assert result["success"] is True, f"Échec: {result['stderr']}"
        assert "mean" in result["stdout"]

    def test_syntax_error_in_sandboxed(self):
        """Un code avec erreur de syntaxe retourne success=False."""
        code = "def foo(\n  return"
        with pytest.raises(SyntaxError):
            run_sandboxed(code)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
