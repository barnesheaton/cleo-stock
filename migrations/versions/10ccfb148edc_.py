"""empty message

Revision ID: 10ccfb148edc
Revises: 320ad5a36262
Create Date: 2022-01-12 12:58:57.702859

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '10ccfb148edc'
down_revision = '320ad5a36262'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('stock_model', sa.Column('date', sa.DateTime(), nullable=True))
    op.add_column('stock_model', sa.Column('description', sa.String(length=128), nullable=True))
    op.add_column('stock_model', sa.Column('name', sa.String(length=128), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('stock_model', 'name')
    op.drop_column('stock_model', 'description')
    op.drop_column('stock_model', 'date')
    # ### end Alembic commands ###
