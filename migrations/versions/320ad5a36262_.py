"""empty message

Revision ID: 320ad5a36262
Revises: eb645245f53a
Create Date: 2022-01-10 19:54:13.346715

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '320ad5a36262'
down_revision = 'eb645245f53a'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('stock_model',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('tickers', sa.ARRAY(sa.Integer()), nullable=True),
    sa.Column('features', sa.Integer(), nullable=True),
    sa.Column('pickle', sa.PickleType(), nullable=True),
    sa.Column('possible_outcomes', sa.JSON(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('stock_model')
    # ### end Alembic commands ###
