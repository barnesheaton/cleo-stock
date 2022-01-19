from wsgiref.validate import validator
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, DecimalField, SelectField
from wtforms.fields.html5 import DateField
from wtforms.validators import DataRequired, ValidationError, Optional

class QueueForm(FlaskForm):
    submit = SubmitField('Queue Task')

class UpdateStockDataForm(FlaskForm):
    start = DecimalField('Start Index', places=0)
    end = DecimalField('End Index', places=0)
    period = StringField('Period')
    submit = SubmitField('Update')

class TrainModelForm(FlaskForm):
    name = StringField(label='Name', validators=[DataRequired()])
    description = StringField(label='Description', render_kw={"placeholder": "A description of the model"})
    tickers = StringField(label="Tickers", render_kw={"placeholder": "A string of tickers"})
    samplePercent = DecimalField(label='Sample Percent', validators=[Optional()], places=0, render_kw={"placeholder": "Percent of DB tickers to sample"})
    submit = SubmitField()

    def validate_tickers(self, field):
        if field.data and self.samplePercent.data:
            raise ValidationError("Only one method of sampling tickers is allowed.")
        if not self.samplePercent.data and not self.tickers.data:
            raise ValidationError("Must choose a method of sampling tickers")

    def validate_samplePercent(self, field):
        if field.data and self.tickers.data:
            raise ValidationError("Only one method of sampling tickers is allowed.")
        if not self.samplePercent.data and not self.tickers.data:
            raise ValidationError("Must choose a method of sampling tickers")

class SimulateForm(FlaskForm):
    model = SelectField(label='model', choices=[])
    start_date = DateField(validators=[DataRequired()])
    end_date = DateField(validators=[DataRequired()])
    principal = DecimalField(label='Starting Principal', places=0, render_kw={"placeholder": "Starting Principal"})
    lookback = DecimalField(label='Lookback Period', places=0 , render_kw={"placeholder": "Lookback Period"})
    lookahead = DecimalField(label='Prediction Period', render_kw={"placeholder": "Prediction Period"})
    diversification = DecimalField(label='Purchasing Spread', render_kw={"placeholder": "Purchasing Spread"})
    submit = SubmitField()