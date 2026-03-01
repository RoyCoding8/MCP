"""Statistics tool — descriptive and inferential statistics."""

import math
from statistics import mean, median, mode, stdev, variance, geometric_mean, harmonic_mean

OPERATIONS = {
    "mean", "median", "mode", "std", "variance",
    "correlation", "regression", "percentile", "zscore", "describe",
    "skewness", "kurtosis", "geometric_mean", "harmonic_mean",
}


def statistics_tool(data: list[float], operation: str = "describe",
                    data_y: list[float] = None, percentile_value: float = None) -> dict:
    """All-in-one statistics tool.

    Use for statistics. Operations: describe, mean, median, mode, std, variance,
    correlation, regression, percentile, zscore, skewness, kurtosis, geometric_mean,
    harmonic_mean. Pass data as array of numbers.
    """
    try:
        if not data:
            return {"error": "data cannot be empty"}

        if operation == "describe":
            m = mean(data)
            s = stdev(data) if len(data) > 1 else 0
            return {
                "count": len(data), "mean": m, "median": median(data),
                "std": s, "variance": s ** 2,
                "min": min(data), "max": max(data), "verified": True,
            }

        elif operation == "mean":
            return {"result": mean(data), "verified": True}

        elif operation == "median":
            return {"result": median(data), "verified": True}

        elif operation == "mode":
            return {"result": mode(data), "verified": True}

        elif operation == "std":
            return {"result": stdev(data), "verified": True}

        elif operation == "variance":
            return {"result": variance(data), "verified": True}

        elif operation == "correlation":
            if not data_y or len(data) != len(data_y):
                return {"error": "correlation requires data_y of same length as data"}
            return {"correlation": _pearson(data, data_y), "verified": True}

        elif operation == "regression":
            if not data_y or len(data) != len(data_y):
                return {"error": "regression requires data_y of same length as data"}
            slope, intercept, r_sq = _linreg(data, data_y)
            return {"slope": slope, "intercept": intercept, "r_squared": r_sq, "verified": True}

        elif operation == "percentile":
            if percentile_value is None:
                return {"error": "percentile_value required (0-100)"}
            return {"result": _percentile(data, percentile_value), "verified": True}

        elif operation == "zscore":
            m, s = mean(data), stdev(data)
            if s == 0:
                return {"error": "std is 0, z-scores undefined"}
            return {"zscores": [(x - m) / s for x in data], "verified": True}

        elif operation == "skewness":
            n = len(data)
            if n < 3:
                return {"error": "skewness requires at least 3 data points"}
            m, s = mean(data), stdev(data)
            if s == 0:
                return {"error": "std is 0, skewness undefined"}
            skew = (n / ((n - 1) * (n - 2))) * sum(((x - m) / s) ** 3 for x in data)
            return {"result": skew, "verified": True}

        elif operation == "kurtosis":
            n = len(data)
            if n < 4:
                return {"error": "kurtosis requires at least 4 data points"}
            m, s = mean(data), stdev(data)
            if s == 0:
                return {"error": "std is 0, kurtosis undefined"}
            kurt = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * sum(((x - m) / s) ** 4 for x in data)
            kurt -= (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
            return {"result": kurt, "verified": True}

        elif operation == "geometric_mean":
            if any(x <= 0 for x in data):
                return {"error": "geometric_mean requires all positive values"}
            return {"result": geometric_mean(data), "verified": True}

        elif operation == "harmonic_mean":
            if any(x <= 0 for x in data):
                return {"error": "harmonic_mean requires all positive values"}
            return {"result": harmonic_mean(data), "verified": True}

        else:
            return {"error": f"Unknown operation '{operation}'. Use: {', '.join(sorted(OPERATIONS))}"}

    except Exception as e:
        return {"error": str(e), "operation": operation}


def _pearson(x, y):
    n = len(x)
    mx, my = mean(x), mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = math.sqrt(sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y))
    return num / den if den else 0


def _linreg(x, y):
    mx, my = mean(x), mean(y)
    ss_xy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    ss_xx = sum((xi - mx) ** 2 for xi in x)
    slope = ss_xy / ss_xx if ss_xx else 0
    intercept = my - slope * mx
    return slope, intercept, _pearson(x, y) ** 2


def _percentile(data, p):
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    f, c = int(k), math.ceil(k)
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)
