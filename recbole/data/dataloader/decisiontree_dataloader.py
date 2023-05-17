
"""
recbole.data.dataloader.decisiontree_dataloader
################################################
"""

from recbole.data.dataloader.general_dataloader import GeneralDataLoader, GeneralNegSampleDataLoader, \
    GeneralFullDataLoader


class DecisionTreeDataLoader(GeneralDataLoader):
    """:class:`DecisionTreeDataLoader` is inherit from
    :class:`~recbole.data.dataloader.general_dataloader.GeneralDataLoader`,
    and didn't add/change anything at all.
    """
    pass


class DecisionTreeNegSampleDataLoader(GeneralNegSampleDataLoader):
    """:class:`DecisionTreeNegSampleDataLoader` is inherit from
    :class:`~recbole.data.dataloader.general_dataloader.GeneralNegSampleDataLoader`,
    and didn't add/change anything at all.
    """
    pass


class DecisionTreeFullDataLoader(GeneralFullDataLoader):
    """:class:`DecisionTreeFullDataLoader` is inherit from
    :class:`~recbole.data.dataloader.general_dataloader.GeneralFullDataLoader`,
    and didn't add/change anything at all.
    """
    pass
