# src/data_validation.py

import great_expectations as gx
import pandas as pd
from typing import Dict, Any

import warnings
warnings.filterwarnings("ignore", message="`result_format` configured at the Validator-level.*")

VALID_CATEGORIES = ["grocery", "restaurant", "retail", "online", "travel"]


def validate_transaction(data: Dict[str, Any]) -> Dict[str, Any]:
    errors = []

    amount = data.get("amount")
    if amount is None:
        errors.append("amount is required")
    elif not isinstance(amount, (int, float)):
        errors.append(f"amount must be a number (got {type(amount).__name__})")
    elif amount <= 0:
        errors.append("amount must be positive")
    elif amount > 50000:
        errors.append(f"amount exceeds maximum allowed value of 50,000 (got {amount:,.2f})")

    hour = data.get("hour")
    if hour is None:
        errors.append("hour is required")
    elif not isinstance(hour, int):
        errors.append(f"hour must be an integer (got {type(hour).__name__})")
    elif not (0 <= hour <= 23):
        errors.append(f"hour must be between 0 and 23 (got {hour})")

    day = data.get("day_of_week")
    if day is None:
        errors.append("day_of_week is required")
    elif not isinstance(day, int):
        errors.append(f"day_of_week must be an integer (got {type(day).__name__})")
    elif not (0 <= day <= 6):
        errors.append(f"day_of_week must be between 0 (Monday) and 6 (Sunday) (got {day})")

    category = data.get("merchant_category")
    if category is None:
        errors.append("merchant_category is required")
    elif not isinstance(category, str):
        errors.append(f"merchant_category must be a string (got {type(category).__name__})")
    elif category not in VALID_CATEGORIES:
        errors.append(
            f"merchant_category must be one of {VALID_CATEGORIES} (got '{category}')"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


def validate_batch(df: pd.DataFrame) -> Dict[str, Any]:
    context = gx.get_context()

    datasource_name = "txn_pandas_source"
    asset_name = "txn_dataframe_asset"
    batch_def_name = "txn_whole_dataframe_batch"
    suite_name = "transaction_suite"

    try:
        ds = context.data_sources.get(datasource_name)
    except Exception:
        ds = context.data_sources.add_pandas(name=datasource_name)

    try:
        asset = ds.get_asset(asset_name)
    except Exception:
        asset = ds.add_dataframe_asset(name=asset_name)

    try:
        batch_definition = asset.get_batch_definition(batch_def_name)
    except Exception:
        batch_definition = asset.add_batch_definition_whole_dataframe(batch_def_name)

    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    validator = context.get_validator(
        batch=batch,
        create_expectation_suite_with_name=suite_name,
    )

    expectation_results = []

    r = validator.expect_column_values_to_not_be_null("amount")
    expectation_results.append(("amount_not_null", r.success, r.result))

    r = validator.expect_column_values_to_be_between(
        "amount", min_value=0.01, max_value=50000, mostly=0.99
    )
    expectation_results.append(("amount_range", r.success, r.result))

    r = validator.expect_column_values_to_not_be_null("hour")
    expectation_results.append(("hour_not_null", r.success, r.result))

    r = validator.expect_column_values_to_be_between("hour", min_value=0, max_value=23)
    expectation_results.append(("hour_range", r.success, r.result))

    r = validator.expect_column_values_to_not_be_null("day_of_week")
    expectation_results.append(("day_of_week_not_null", r.success, r.result))

    r = validator.expect_column_values_to_be_between(
        "day_of_week", min_value=0, max_value=6
    )
    expectation_results.append(("day_range", r.success, r.result))

    r = validator.expect_column_values_to_not_be_null("merchant_category")
    expectation_results.append(("merchant_category_not_null", r.success, r.result))

    r = validator.expect_column_values_to_be_in_set(
        "merchant_category", VALID_CATEGORIES
    )
    expectation_results.append(("category_valid", r.success, r.result))

    validation_result = validator.validate()

    passed = sum(1 for _, success, _ in expectation_results if success)
    total = len(expectation_results)

    return {
        "success": validation_result.success,
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total else 0.0,
        "details": {
            name: {"passed": success, "result": result}
            for name, success, result in expectation_results
        },
        "validation_result": validation_result,
    }

if __name__ == "__main__":
    print("-"*60)
    print("TESTING DATA VALIDATION")
    print("-"*60)
    
    # Test single transaction validation
    print("\n1. Single Transaction Validation")
    print("-"*40)
    
    test_cases = [
        {
            "name": "Valid transaction",
            "data": {"amount": 50.0, "hour": 14, "day_of_week": 3, "merchant_category": "grocery"}
        },
        {
            "name": "Negative amount",
            "data": {"amount": -100.0, "hour": 14, "day_of_week": 3, "merchant_category": "grocery"}
        },
        {
            "name": "Invalid hour",
            "data": {"amount": 50.0, "hour": 25, "day_of_week": 3, "merchant_category": "grocery"}
        },
        {
            "name": "Unknown merchant",
            "data": {"amount": 50.0, "hour": 14, "day_of_week": 3, "merchant_category": "unknown"}
        },
        {
            "name": "Everything wrong",
            "data": {"amount": -999, "hour": 99, "day_of_week": 15, "merchant_category": "fake"}
        },
    ]
    
    for tc in test_cases:
        result = validate_transaction(tc["data"])
        status = "PASS" if result["valid"] else "FAIL"
        print(f"\n{tc['name']}: {status}")
        if result["errors"]:
            for error in result["errors"]:
                print(f"  - {error}")
    
    # Test batch validation
    print("\n\n2. Batch Validation with Great Expectations")
    print("-"*40)
    
    train_df = pd.read_csv('data/train.csv')
    results = validate_batch(train_df)
    
    print(f"\nTraining data validation: {results['passed']}/{results['total']} checks passed")
    print(f"Pass rate: {results['pass_rate']:.1%}")
    
    if not results['success']:
        print("\nFailed checks:")
        for name, detail in results['details'].items():
            if not detail['passed']:
                print(f"  - {name}")