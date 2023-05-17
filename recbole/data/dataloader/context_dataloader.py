
"""
recbole.data.dataloader.context_dataloader
################################################
"""

from recbole.data.dataloader.general_dataloader import GeneralDataLoader, GeneralNegSampleDataLoader, \
    GeneralFullDataLoader


class ContextDataLoader(GeneralDataLoader):
    """:class:`ContextDataLoader` is inherit from
    :class:`~recbole.data.dataloader.general_dataloader.GeneralDataLoader`,
    and didn't add/change anything at all.
    """
    pass


class ContextNegSampleDataLoader(GeneralNegSampleDataLoader):
    """:class:`ContextNegSampleDataLoader` is inherit from
    :class:`~recbole.data.dataloader.general_dataloader.GeneralNegSampleDataLoader`,
    and didn't add/change anything at all.
    """
    pass


class ContextFullDataLoader(GeneralFullDataLoader):
    """:class:`ContextFullDataLoader` is inherit from
    :class:`~recbole.data.dataloader.general_dataloader.GeneralFullDataLoader`,
    and didn't add/change anything at all.
    """
    pass
