"""
Input validation for TxGemma MCP server.

Validates user inputs to prevent injection attacks, resource exhaustion,
and other security issues.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when input validation fails."""

    pass


class InputValidator:
    """
    Validates user inputs for TxGemma tools.

    Prevents:
    - Excessively long inputs (DoS)
    - Special characters in tool names (injection)
    - Invalid parameter names
    - Malformed data types
    """

    # Maximum lengths to prevent DoS
    MAX_TOOL_NAME_LENGTH = 100
    MAX_PARAM_NAME_LENGTH = 100
    MAX_STRING_VALUE_LENGTH = 50000
    MAX_TOTAL_PARAMS = 10

    TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
    PARAM_NAME_PATTERN = re.compile(r"^[a-z_][a-z0-9_]*$")

    @classmethod
    def validate_tool_name(cls, tool_name: str) -> str:
        """
        Validate tool name.

        Args:
            tool_name: Tool name to validate

        Returns:
            Validated tool name

        Raises:
            ValidationError: If tool name is invalid
        """
        if not isinstance(tool_name, str):
            raise ValidationError(f"Tool name must be string, got {type(tool_name)}")

        if not tool_name:
            raise ValidationError("Tool name cannot be empty")

        if len(tool_name) > cls.MAX_TOOL_NAME_LENGTH:
            raise ValidationError(
                f"Tool name too long: {len(tool_name)} > {cls.MAX_TOOL_NAME_LENGTH}"
            )

        if not cls.TOOL_NAME_PATTERN.match(tool_name):
            raise ValidationError(
                f"Invalid tool name: {tool_name}. Must contain only alphanumeric, underscore, hyphen"
            )

        return tool_name

    @classmethod
    def validate_param_name(cls, param_name: str) -> str:
        """
        Validate parameter name.

        Args:
            param_name: Parameter name to validate

        Returns:
            Validated parameter name

        Raises:
            ValidationError: If parameter name is invalid
        """
        if not isinstance(param_name, str):
            raise ValidationError(f"Parameter name must be string, got {type(param_name)}")

        if not param_name:
            raise ValidationError("Parameter name cannot be empty")

        if len(param_name) > cls.MAX_PARAM_NAME_LENGTH:
            raise ValidationError(
                f"Parameter name too long: {len(param_name)} > {cls.MAX_PARAM_NAME_LENGTH}"
            )

        if not cls.PARAM_NAME_PATTERN.match(param_name):
            raise ValidationError(f"Invalid parameter name: {param_name}. Must be snake_case")

        return param_name

    @classmethod
    def validate_string_value(cls, value: str, param_name: str = "value") -> str:
        """
        Validate string parameter value.

        Args:
            value: String value to validate
            param_name: Name of parameter (for error messages)

        Returns:
            Validated string value

        Raises:
            ValidationError: If value is invalid
        """
        if not isinstance(value, str):
            raise ValidationError(f"Parameter '{param_name}' must be string, got {type(value)}")

        if len(value) > cls.MAX_STRING_VALUE_LENGTH:
            raise ValidationError(
                f"Parameter '{param_name}' too long: {len(value)} > {cls.MAX_STRING_VALUE_LENGTH}"
            )

        # Check for null bytes (could cause issues in string processing)
        if "\x00" in value:
            raise ValidationError(f"Parameter '{param_name}' contains null bytes")

        return value

    @classmethod
    def validate_arguments(cls, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Validate all arguments for a tool call.

        Args:
            arguments: Dictionary of parameter name -> value

        Returns:
            Validated arguments dictionary

        Raises:
            ValidationError: If any argument is invalid
        """
        if not isinstance(arguments, dict):
            raise ValidationError(f"Arguments must be dict, got {type(arguments)}")

        if len(arguments) > cls.MAX_TOTAL_PARAMS:
            raise ValidationError(f"Too many parameters: {len(arguments)} > {cls.MAX_TOTAL_PARAMS}")

        validated = {}

        for param_name, value in arguments.items():
            # Validate parameter name
            cls.validate_param_name(param_name)

            # Validate value based on type
            if isinstance(value, str):
                validated[param_name] = cls.validate_string_value(value, param_name)
            elif isinstance(value, (int, float)):
                # Numbers are safe
                validated[param_name] = value
            elif isinstance(value, bool):
                # Booleans are safe
                validated[param_name] = value
            elif value is None:
                # None is safe (optional param)
                validated[param_name] = value
            else:
                raise ValidationError(
                    f"Parameter '{param_name}' has unsupported type: {type(value)}"
                )

        return validated

    @classmethod
    def validate_tool_call(cls, tool_name: str, arguments: dict[str, Any]) -> tuple[str, dict]:
        """
        Validate complete tool call.

        Args:
            tool_name: Tool name
            arguments: Tool arguments

        Returns:
            Tuple of (validated_tool_name, validated_arguments)

        Raises:
            ValidationError: If validation fails
        """
        try:
            validated_name = cls.validate_tool_name(tool_name)
            validated_args = cls.validate_arguments(arguments)

            logger.debug(
                f"Validation passed for tool '{validated_name}' "
                f"with {len(validated_args)} parameters"
            )

            return validated_name, validated_args

        except ValidationError as e:
            logger.warning(f"Validation failed for tool '{tool_name}': {e}")
            raise


def validate_tool_call(tool_name: str, arguments: dict[str, Any]) -> tuple[str, dict]:
    """
    Validate a tool call.

    This is a convenience wrapper around InputValidator.validate_tool_call().

    Args:
        tool_name: Tool name to validate
        arguments: Tool arguments to validate

    Returns:
        Tuple of (validated_tool_name, validated_arguments)

    Raises:
        ValidationError: If validation fails
    """
    return InputValidator.validate_tool_call(tool_name, arguments)
