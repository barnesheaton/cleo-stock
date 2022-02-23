from wsgiref.validate import validator
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, DecimalField, SelectField, RadioField
from wtforms.fields.html5 import DateField
from wtforms.validators import DataRequired, ValidationError, Optional

class QueueForm(FlaskForm):
    submit = SubmitField('Queue Task')

class UpdateStockDataForm(FlaskForm):
    start = DecimalField('Start Index', places=0, render_kw={"placeholder": "0"})
    end = DecimalField('End Index', places=0, render_kw={"placeholder": "4000"})
    period = StringField('Period', render_kw={"placeholder": "1yr, 1mo, max"})
    submit = SubmitField('Update')

class TrainModelForm(FlaskForm):
    name = StringField(label='Name', validators=[DataRequired()])
    description = StringField(label='Description', render_kw={"placeholder": "A description of the model"})
    tickers = StringField(label="Tickers", render_kw={"placeholder": "A string of tickers"})
    sample_percent = DecimalField(label='Sample Percent', validators=[Optional()], places=0, render_kw={"placeholder": "Percent of DB tickers to sample"})
    observation_period = DecimalField(label='Observation Period', places=0)
    model_type = RadioField(label='Observation Period', choices=['pomegranate', 'default'])
    submit = SubmitField()

    def validate_tickers(self, field):
        if field.data and self.sample_percent.data:
            raise ValidationError("Only one method of sampling tickers is allowed.")
        if not self.sample_percent.data and not self.tickers.data:
            raise ValidationError("Must choose a method of sampling tickers")

    def validate_sample_percent(self, field):
        if field.data and self.tickers.data:
            raise ValidationError("Only one method of sampling tickers is allowed.")
        if not self.sample_percent.data and not self.tickers.data:
            raise ValidationError("Must choose a method of sampling tickers")

class SimulateForm(FlaskForm):
    model = SelectField(label='model', choices=[])
    start_date = DateField(validators=[DataRequired()])
    end_date = DateField(validators=[DataRequired()])
    principal = DecimalField(label='Starting Principal', places=0, render_kw={"placeholder": "Starting Principal"})
    lookback = DecimalField(label='Lookback Period', places=0 , render_kw={"placeholder": "Lookback Period"})
    lookahead = DecimalField(label='Prediction Period', render_kw={"placeholder": "Prediction Period"})
    diversification = DecimalField(label='Purchasing Spread', validators=[Optional()], render_kw={"placeholder": "Purchasing Spread"})
    submit = SubmitField()

class PlotForm(FlaskForm):
    model = SelectField(label='model', choices=[])
    tickers = StringField(label="Tickers", render_kw={"placeholder": "A string of tickers"})
    lookback = DecimalField(label='Lookback Period', places=0 , render_kw={"placeholder": "Lookback Period"})
    lookahead = DecimalField(label='Prediction Period', render_kw={"placeholder": "Prediction Period"})
    limit = DecimalField(label='Ticker Data Limit', validators=[Optional()], render_kw={"placeholder": "optional limit on Ticker Data"})
    submit = SubmitField()

class DisplayPlotForm(FlaskForm):
    task = SelectField(label='task', choices=[])
    submit = SubmitField()