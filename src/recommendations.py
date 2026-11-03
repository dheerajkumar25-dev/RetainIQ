"""Rule-based retention considerations shown next to a prediction.

These are conversation starters for a customer-success team, not guaranteed
fixes: each rule fires only when the model's own explanation says the factor
pushes this customer towards churn.
"""
import pandas as pd

HIGH_MONTHLY_CHARGE = 70.0
LOW_TENURE_MONTHS = 12


def _rule_matches(feature: str, value, customer: dict) -> str | None:
    internet = customer.get("InternetService")
    if feature == "Contract" and value == "Month-to-month":
        return "Month-to-month customers can leave at any time. Consider offering an incentive to move to a one- or two-year plan."
    if feature == "tenure" and value <= LOW_TENURE_MONTHS:
        return "The customer is early in their lifecycle. A proactive onboarding check-in may help surface problems early."
    if feature == "TechSupport" and value == "No" and internet != "No":
        return "No technical support add-on. Offering a trial of support could reduce frustration with service issues."
    if feature == "OnlineSecurity" and value == "No" and internet != "No":
        return "No online security add-on. A bundled security offer may increase the value the customer sees in the plan."
    if feature == "InternetService" and value == "Fiber optic":
        return "Fiber customers churned more often in this dataset. Review price-to-value and service quality for this account."
    if feature == "PaymentMethod" and value == "Electronic check":
        return "Electronic-check payers churned more often here. Encouraging automatic payment may reduce friction."
    if feature == "MonthlyCharges" and value >= HIGH_MONTHLY_CHARGE:
        return "The monthly bill is relatively high. Check whether a better-fitting plan or bundle would suit the customer."
    return None


def suggest_retention_actions(customer: dict, drivers: pd.DataFrame, max_items: int = 3) -> list[str]:
    """Return up to `max_items` considerations based on the top risk-increasing drivers."""
    suggestions = []
    increasing = drivers[drivers["shap_value"] > 0].sort_values("shap_value", ascending=False)
    for _, row in increasing.iterrows():
        text = _rule_matches(row["feature"], row["value"], customer)
        if text and text not in suggestions:
            suggestions.append(text)
        if len(suggestions) == max_items:
            break
    return suggestions
