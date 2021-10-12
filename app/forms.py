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
