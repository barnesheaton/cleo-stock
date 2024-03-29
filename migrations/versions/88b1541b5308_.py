"""empty message

Revision ID: 88b1541b5308
Revises: 10ccfb148edc
Create Date: 2022-01-15 09:53:48.431084

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '88b1541b5308'
down_revision = '10ccfb148edc'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('simulation',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('model_id', sa.Integer(), nullable=True),
    sa.Column('date', sa.DateTime(), nullable=True),
    sa.Column('start_date', sa.Date(), nullable=True),
    sa.Column('end_date', sa.Date(), nullable=True),
    sa.Column('simulation_period', sa.Integer(), nullable=True),
    sa.Column('close_accuracy', sa.Float(), nullable=True),
    sa.Column('open_accuracy', sa.Float(), nullable=True),
    sa.Column('total_accuracy', sa.Float(), nullable=True),
    sa.Column('starting_capital', sa.Float(), nullable=True),
    sa.Column('ending_capital', sa.Float(), nullable=True),
    sa.Column('complete', sa.Boolean(), nullable=False),    
    sa.ForeignKeyConstraint(['model_id'], ['stock_model.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    op.drop_table('simulation')
    # ### end Alembic commands ###
