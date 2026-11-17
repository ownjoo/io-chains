"""
One error-handling contract, shared by every link that can fail per item
(Processor, Enricher, PersistenceLink).

When processing an item raises, in order:
  1. error_subscribers (graph error edges) receive an ErrorEnvelope. If any are
     wired, the edge owns the item: it is not published downstream and
     on_error is NOT called.
  2. on_error_event(envelope) fires — always, for observability only.
  3. on_error(exc, item) is called if no edge handled it:
       - returns None, Skip(), or the Skip class  -> item dropped (items_skipped += 1)
       - returns any other value                  -> published in place of the item
       - raises                                   -> that exception propagates
  4. Neither wired -> the original exception propagates.

items_errored increments once per failed item, whichever path handles it.
"""

import asyncio
import unittest

from io_chains import Collector, Enricher, PersistenceLink, Processor, Relation, Skip
from io_chains._internal.sentinel import END_OF_STREAM, ErrorEnvelope

ITEMS = [{'id': 1}, {'id': 2}, {'id': 3}]  # id 2 always fails


class Boom(Exception):
    pass


def _fail_on_2(item):
    if item['id'] == 2:
        raise Boom('id 2')
    return item


class _FakeManager:
    async def aupsert(self, store_id, key, value):
        if key == '2':
            raise Boom('write failed')


def _key_fail_on_2(key):
    if key == 2:
        raise Boom('lookup failed')
    return key


async def _run_processor(edge=None, **kw):
    out = Collector()
    link = Processor(source=ITEMS, processor=_fail_on_2, subscribers=out, **kw)
    link.error_subscribers = edge
    await link()
    return link, out


async def _run_persistence(edge=None, **kw):
    out = Collector()
    link = PersistenceLink(
        source=ITEMS, manager=_FakeManager(), store_id='s', key_fn=lambda x: str(x['id']), subscribers=out, **kw
    )
    link.error_subscribers = edge
    await link()
    return link, out


async def _run_enricher(edge=None, **kw):
    out = Collector()
    link = Enricher(
        primary_channel='p',
        subscribers=out,
        relations=[Relation(from_field='id', to_channel='s', to_field='id', attach_as='s', key_transform=_key_fail_on_2)],
        **kw,
    )
    link.error_subscribers = edge
    primary = Processor(source=ITEMS)
    side = Processor(source=[{'id': 1}, {'id': 3}])
    primary.subscribe(link, channel='p')
    side.subscribe(link, channel='s')
    await asyncio.gather(primary(), side(), link())
    return link, out


RUNNERS = {'Processor': _run_processor, 'PersistenceLink': _run_persistence, 'Enricher': _run_enricher}


class TestOnErrorContract(unittest.IsolatedAsyncioTestCase):
    async def _each(self, check):
        for name, run in RUNNERS.items():
            with self.subTest(link=name):
                await check(run)

    @staticmethod
    def _ids(out):
        return [item['id'] for item in out]

    async def test_unhandled_error_propagates(self):
        async def check(run):
            with self.assertRaises(Boom):
                await run()

        await self._each(check)

    async def test_returning_none_drops_item(self):
        async def check(run):
            link, out = await run(on_error=lambda e, item: None)
            self.assertEqual(self._ids(out), [1, 3])
            self.assertEqual(link.items_errored, 1)
            self.assertEqual(link.items_skipped, 1)

        await self._each(check)

    async def test_returning_skip_instance_drops_item(self):
        async def check(run):
            link, out = await run(on_error=lambda e, item: Skip())
            self.assertEqual(self._ids(out), [1, 3])
            self.assertEqual(link.items_skipped, 1)

        await self._each(check)

    async def test_returning_skip_class_drops_item(self):
        """`return Skip` (no parens) is an easy slip — it must not leak downstream."""

        async def check(run):
            link, out = await run(on_error=lambda e, item: Skip)
            self.assertEqual(self._ids(out), [1, 3])
            self.assertEqual(link.items_skipped, 1)

        await self._each(check)

    async def test_returning_value_replaces_item(self):
        async def check(run):
            link, out = await run(on_error=lambda e, item: {'id': 99})
            self.assertEqual(self._ids(out), [1, 99, 3])
            self.assertEqual(link.items_errored, 1)
            self.assertEqual(link.items_skipped, 0)

        await self._each(check)

    async def test_returning_the_item_passes_it_through(self):
        async def check(run):
            _, out = await run(on_error=lambda e, item: item)
            self.assertEqual(self._ids(out), [1, 2, 3])

        await self._each(check)

    async def test_async_on_error_is_awaited(self):
        async def handler(e, item):
            await asyncio.sleep(0)
            return {'id': 99}

        async def check(run):
            _, out = await run(on_error=handler)
            self.assertEqual(self._ids(out), [1, 99, 3])

        await self._each(check)

    async def test_on_error_receives_exception_and_original_item(self):
        async def check(run):
            seen = []
            await run(on_error=lambda e, item: seen.append((type(e), item['id'])))
            self.assertEqual(seen, [(Boom, 2)])

        await self._each(check)

    async def test_exception_raised_by_on_error_propagates(self):
        def handler(e, item):
            raise RuntimeError('handler gave up')

        async def check(run):
            with self.assertRaises(RuntimeError):
                await run(on_error=handler)

        await self._each(check)

    async def test_error_edge_takes_precedence_over_on_error(self):
        async def check(run):
            edge = Collector()
            called = []
            link, out = await run(edge=edge, on_error=lambda e, item: called.append(item) or item)
            await edge.push(END_OF_STREAM)
            envelopes = list(edge)
            self.assertEqual(len(envelopes), 1)
            self.assertIsInstance(envelopes[0], ErrorEnvelope)
            self.assertEqual(envelopes[0].datum['id'], 2)
            self.assertTrue(envelopes[0].handled)
            self.assertEqual(called, [])
            self.assertEqual(self._ids(out), [1, 3])
            self.assertEqual(link.items_errored, 1)

        await self._each(check)

    async def test_on_error_event_fires_once_alongside_on_error(self):
        async def check(run):
            events = []
            await run(on_error=lambda e, item: None, on_error_event=events.append)
            self.assertEqual(len(events), 1)
            self.assertIsInstance(events[0].exc, Boom)
            self.assertFalse(events[0].handled)

        await self._each(check)

    async def test_on_error_event_fires_before_unhandled_raise(self):
        async def check(run):
            events = []
            with self.assertRaises(Boom):
                await run(on_error_event=events.append)
            self.assertEqual(len(events), 1)

        await self._each(check)



class TestSkipClassFromProcessorFunction(unittest.IsolatedAsyncioTestCase):
    async def test_processor_returning_skip_class_drops_item(self):
        out = Collector()
        p = Processor(source=[1, 2], processor=lambda x: Skip if x == 1 else x, subscribers=out)
        await p()
        self.assertEqual(list(out), [2])
        self.assertEqual(p.items_skipped, 1)


if __name__ == '__main__':
    unittest.main()
