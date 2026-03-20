Feature Selection Strategy for Air Quality Forecasting in Almaty

To optimize the Prophet forecasting model for Almaty’s unique geographical conditions (mountain basin and “bowl effect”), each pollutant is modeled separately using carefully selected meteorological and temporal regressors.

Almaty’s topography traps polluted air, especially during temperature inversions in winter. Therefore, incorporating weather variables transforms the model from purely statistical to physics-informed forecasting.

1. PM2.5 (Fine Particulate Matter)

PM2.5 is the most critical pollutant in Almaty. Its concentration is strongly influenced by combustion sources and atmospheric trapping conditions.

Mandatory Features

wind_speed

temp

humidity

Justification

Wind Speed → Primary dispersal mechanism in the Almaty “bowl.”

Humidity → Increases particle mass through condensation.

Low Temperature → Triggers higher emissions due to residential heating.

Additional Regressor

no2 (proxy for traffic intensity)

Traffic is a major contributor to PM2.5. Since NO₂ strongly correlates with vehicle activity, it improves predictive performance.

Seasonality Priority

Yearly

Daily

2. PM10 (Coarse Particulate Matter)

PM10 consists of larger particles such as dust, soot, and road debris.

Mandatory Features

wind_speed

precipitation (if available)

The "Wind Twist"

Unlike fine particles and gases, very high wind speeds can increase PM10 by re-suspending dust from the ground.

Justification

PM10 behavior is more mechanical than chemical. It reacts strongly to:

Atmospheric turbulence

Surface dust movement

Physical air disturbance

Seasonality Priority

Yearly

3. NO₂ (Nitrogen Dioxide)

NO₂ is primarily an indicator of traffic emissions.

Mandatory Features

temp

is_weekend or day_of_week

Justification

NO₂ follows a strict two-peak daily cycle:

Morning rush (~08:00)

Evening rush (~18:00)

Cold engines emit more pollutants → temperature effect.

Prophet Configuration
daily_seasonality=True


This is required to capture rush-hour spikes.

Seasonality Priority

Daily

Weekly

4. SO₂ (Sulfur Dioxide)

SO₂ acts as a Coal Combustion Indicator in Almaty.

Mandatory Features

temp

Justification

Originates mainly from:

Thermal Power Plants (CHPs)

Residential coal burning

Strong negative correlation with temperature

Nearly disappears in summer when heating is off

Prophet Configuration
yearly_seasonality=True

Seasonality Priority

Yearly

Summary Table
Model Target	Primary Regressor	Secondary Regressor	Seasonality Priority
PM2.5	wind_speed	humidity	Yearly & Daily
PM10	wind_speed	temp	Yearly
NO₂	day_of_week	temp	Daily & Weekly
SO₂	temp	wind_speed	Yearly