from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, DecimalField
from wtforms.fields.html5 import DateField
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
    name = StringField('Name')
    description = StringField('Description')

class SimulateForm(FlaskForm):
    start_date = DateField(validators=[DataRequired()])
    end_date = DateField(validators=[DataRequired()])
    principal = DecimalField(label='Starting Principal', places=0, render_kw={"placeholder": "Starting Principal"})
    lookback = DecimalField(label='Lookback Period', places=0 , render_kw={"placeholder": "Lookback Period"})
    lookahead = DecimalField(label='Prediction Period', render_kw={"placeholder": "Prediction Period"})
    diversification = DecimalField(label='Purchasing Spread', render_kw={"placeholder": "Purchasing Spread"})
    submit = SubmitField()