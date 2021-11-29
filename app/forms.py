from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, DecimalField
from wtforms.validators import DataRequired

class QueueForm(FlaskForm):
    submit = SubmitField('Queue Task')

class UpdateStockDataForm(FlaskForm):
    start = DecimalField('Start Index', places=0)
    end = DecimalField('End Index')
    period = StringField('Period')
    submit = SubmitField('Update')

class TrainModelForm(FlaskForm):
    submit = SubmitField('Train Model')

class SimulateForm(FlaskForm):
    start_date = StringField(label='Start Date', render_kw={"placeholder": "Start Date"})
    end_date = StringField(label='End Date', render_kw={"placeholder": "End Date"})
    principal = DecimalField(label='Starting Principal', render_kw={"placeholder": "Starting Principal"})
    lookback = DecimalField(label='Lookback Period', render_kw={"placeholder": "Lookback Period"})
    lookahead = DecimalField(label='Prediction Period', render_kw={"placeholder": "Prediction Period"})
    diversification = DecimalField(label='Purchasing Spread', render_kw={"placeholder": "Purchasing Spread"})
    submit = SubmitField(label=None)