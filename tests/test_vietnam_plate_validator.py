"""T5 Vietnam plate validator regression tests."""

from __future__ import annotations

from core.vietnam_plate_validator import INVALID, UNCERTAIN, VALID, VietnamPlateValidator


def test_empty_and_obvious_non_plates_are_invalid() -> None:
    validator = VietnamPlateValidator()
    for text in ("", "1", "11", "1111", "CONPHONG", "GIUXE", "00PA01"):
        result = validator.validate(text)
        assert result.status == INVALID


def test_common_car_and_motorcycle_shapes_are_valid() -> None:
    validator = VietnamPlateValidator()
    assert validator.validate("30F05148").status == VALID
    assert validator.validate("59S120468").status == VALID
    assert validator.validate("29HA00233").status == VALID


def test_separator_normalization_is_formatting_only() -> None:
    validator = VietnamPlateValidator()
    result = validator.validate("59-S1 204.68")
    assert result.cleaned_text == "59S120468"
    assert result.normalized_text == "59S120468"
    assert result.corrections == ()


def test_position_aware_o_zero_correction_is_recorded() -> None:
    validator = VietnamPlateValidator()
    result = validator.validate("59S12O468")
    assert result.status == VALID
    assert result.normalized_text == "59S120468"
    assert "O -> 0" in result.corrections


def test_position_aware_i_one_correction_is_not_global() -> None:
    validator = VietnamPlateValidator()
    numeric = validator.validate("59S12I468")
    assert numeric.normalized_text == "59S121468"
    assert "I -> 1" in numeric.corrections

    letter_position = validator.validate("59I120468")
    assert letter_position.normalized_text == "59I120468"
    assert letter_position.corrections == ()


def test_b_eight_is_not_changed_when_letter_position_is_expected() -> None:
    result = VietnamPlateValidator().validate("59B120468")
    assert result.normalized_text == "59B120468"
    assert result.corrections == ()


def test_one_character_shape_damage_is_uncertain() -> None:
    # Ten characters is one longer than the supported nine-character
    # motorcycle form; it is close enough to retain for conservative review.
    result = VietnamPlateValidator().validate("59S1204680")
    assert result.status == UNCERTAIN


def test_unknown_but_plausible_prefix_is_uncertain_not_invalid() -> None:
    result = VietnamPlateValidator().validate("13A12345")
    assert result.status == UNCERTAIN


def test_validator_is_deterministic() -> None:
    validator = VietnamPlateValidator()
    first = validator.validate("59S12O468")
    for _ in range(10):
        assert validator.validate("59S12O468") == first


if __name__ == "__main__":
    for test in (
        test_empty_and_obvious_non_plates_are_invalid,
        test_common_car_and_motorcycle_shapes_are_valid,
        test_separator_normalization_is_formatting_only,
        test_position_aware_o_zero_correction_is_recorded,
        test_position_aware_i_one_correction_is_not_global,
        test_b_eight_is_not_changed_when_letter_position_is_expected,
        test_one_character_shape_damage_is_uncertain,
        test_unknown_but_plausible_prefix_is_uncertain_not_invalid,
        test_validator_is_deterministic,
    ):
        test()
    print("T5 VietnamPlateValidator tests: OK")
