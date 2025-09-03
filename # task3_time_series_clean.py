
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# try to import statsmodels and show version
try:
    import statsmodels as sm
    sm_version = getattr(sm, "__version__", "unknown")
except Exception:
    sm = None
    sm_version = "not installed"

print("Python:", sys.version.splitlines()[0])
print("statsmodels version:", sm_version)

# Try importing ARIMA from multiple locations, else fall back to SARIMAX
ARIMA = None
SARIMAX = None
import_error_msgs = []

try:
    # modern import (statsmodels >= 0.12)
    from statsmodels.tsa.arima.model import ARIMA
    print("Imported ARIMA from statsmodels.tsa.arima.model")
except Exception as e:
    import_error_msgs.append(("statsmodels.tsa.arima.model", str(e)))
    try:
        # legacy import (older statsmodels)
        from statsmodels.tsa.arima_model import ARIMA
        print("Imported ARIMA from statsmodels.tsa.arima_model (legacy)")
    except Exception as e2:
        import_error_msgs.append(("statsmodels.tsa.arima_model", str(e2)))
        try:
            # SARIMAX is widely available and works as a fallback
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            SARIMAX = SARIMAX
            print("Imported SARIMAX from statsmodels.tsa.statespace.sarimax (fallback)")
        except Exception as e3:
            import_error_msgs.append(("statsmodels.tsa.statespace.sarimax", str(e3)))

if import_error_msgs:
    print("\nImport attempts and errors (if any):")
    for loc, msg in import_error_msgs:
        print(f" - {loc}: {msg}")

# If statsmodels not installed or neither ARIMA nor SARIMAX available, instruct user
if ARIMA is None and SARIMAX is None:
    print("\nERROR: Neither ARIMA nor SARIMAX could be imported from statsmodels.")
    print("If statsmodels is not installed, install/upgrade it with one of the following commands:")
    print("  pip install --upgrade statsmodels")
    print("or (conda):")
    print("  conda install -c conda-forge statsmodels")
    raise SystemExit("Cannot proceed: statsmodels ARIMA/SARIMAX is unavailable.")

# load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
df.index = pd.to_datetime(df.index)
df = df.asfreq('MS')  # ensure monthly freq
print("\nLoaded AirPassengers dataset:", df.shape)
print(df.head())

# quick missing check
print("\nMissing values:", df.isnull().sum().to_dict())
df['Passengers'] = df['Passengers'].interpolate()  # safe fill if any

# simple plot
sns.set()
plt.figure(figsize=(10,4))
plt.plot(df.index, df['Passengers'])
plt.title("Monthly Air Passengers (1949-1960)")
plt.show()

# rolling stats
plt.figure(figsize=(10,4))
plt.plot(df['Passengers'], label='Original')
plt.plot(df['Passengers'].rolling(12).mean(), label='12-mo rolling mean')
plt.plot(df['Passengers'].rolling(12).std(), label='12-mo rolling std')
plt.legend()
plt.show()

# train/test split (last 12 months as test)
train = df.iloc[:-12]['Passengers']
test = df.iloc[-12:]['Passengers']

# Build & fit model (choose ARIMA if available else SARIMAX)
model_fit = None
model_type_used = None
order = (2, 1, 2)  # example order; adjust/tune if desired

if ARIMA is not None:
    try:
        # works for both modern and legacy ARIMA (API is similar for simple use)
        model = ARIMA(train, order=order)
        # fit may or may not accept disp argument depending on version
        try:
            model_fit = model.fit()
        except TypeError:
            model_fit = model.fit(disp=False)
        model_type_used = 'ARIMA'
        print(f"\nFitted ARIMA(order={order}) model.")
    except Exception as e:
        print("ARIMA fit failed with error:", e)
        # fall back to SARIMAX if available
        if SARIMAX is not None:
            try:
                model = SARIMAX(train, order=order, seasonal_order=(0,0,0,0))
                model_fit = model.fit(disp=False)
                model_type_used = 'SARIMAX'
                print("\nFitted SARIMAX fallback model.")
            except Exception as e2:
                print("SARIMAX fallback also failed:", e2)
                raise
        else:
            raise
else:
    # ARIMA not available — use SARIMAX
    try:
        model = SARIMAX(train, order=order, seasonal_order=(0,0,0,0))
        model_fit = model.fit(disp=False)
        model_type_used = 'SARIMAX'
        print(f"\nFitted SARIMAX(order={order}) model.")
    except Exception as e:
        print("SARIMAX fit failed with error:", e)
        raise

# show summary if available
try:
    print(model_fit.summary())
except Exception:
    print("Model summary not available for this results object.")

# Forecasting: try to use .forecast(steps) then fallback to get_forecast().predicted_mean
try:
    forecast_vals = model_fit.forecast(steps=12)
except Exception:
    try:
        forecast_vals = model_fit.get_forecast(steps=12).predicted_mean
    except Exception as e:
        print("Forecasting failed:", e)
        raise

# Ensure forecast is a pandas Series with matching index
forecast = pd.Series(forecast_vals, index=test.index)

# Plot train, test, forecast
plt.figure(figsize=(10,5))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(forecast.index, forecast, label='Forecast')
plt.legend()
plt.title(f"Forecast using {model_type_used}")
plt.show()

# Evaluate
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(test, forecast))
print(f"RMSE on last 12 months: {rmse:.3f}")

# Save forecast to CSV
out = pd.DataFrame({'actual': test, 'forecast': forecast})
out.to_csv("airpassengers_forecast_results.csv")
print("Saved forecast results to airpassengers_forecast_results.csv")
