"""empty message

Revision ID: eb645245f53a
Revises: 
Create Date: 2021-02-13 08:52:18.952810

"""
from alembic import op
from sqlalchemy.engine.reflection import Inspector
import sqlalchemy as sa
import app.main.utils as utils


# revision identifiers, used by Alembic.
revision = 'eb645245f53a'
down_revision = None
branch_labels = None
depends_on = None

def hasTable(table_name):
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    tables = inspector.get_table_names()
    if table_name in tables:
        return True
    return False

def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('task',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('name', sa.String(length=128), nullable=True),
    sa.Column('description', sa.String(length=128), nullable=True),
    sa.Column('complete', sa.Boolean(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_task_name'), 'task', ['name'], unique=False)
    # ### end Alembic commands ###

    (ticker_list, _) = utils.getTickerList(start=30, end=40)
    for ticker in ticker_list:
        table_name = ticker.lower()
        op.create_table(table_name,
            sa.Column('date', sa.Date(), primary_key=True, index=True, unique=True),
            sa.Column('open', sa.Float()),
            sa.Column('high', sa.Float()),
            sa.Column('low', sa.Float()),
            sa.Column('close', sa.Float()),
            sa.Column('adj_close', sa.Float()),
            sa.Column('volume', sa.Float()),
        )



def downgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_task_name'), table_name='task')
    op.drop_table('task')
    # ### end Alembic commands ###

    (ticker_list, _) = utils.getTickerList(start=30, end=40)
    for ticker in ticker_list:
        table_name = ticker.lower()
        if hasTable(table_name):
            op.drop_index(op.f(f"ix_{table_name}_date"), table_name=table_name)
            op.drop_table(table_name)
