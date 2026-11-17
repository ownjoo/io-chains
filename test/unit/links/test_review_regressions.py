"""
Regression tests for bugs found in the 2026-11 code review.

  1. Chain.subscribe() / Chain.subscribers= after construction were silently
     dropped (wired to the Chain's own, never-published subscriber list).
  2. A cancelled link scheduled an unreferenced EOS task that could block
     forever on a full bounded downstream queue.
  3. Enricher keyed lookups by channel only, so two Relations on the same
     channel with different to_field clobbered each other.
  4. Processor counted items_errored once per failed *attempt*, so an item
     that eventually succeeded after retries still showed as errored.
  5. Publisher fan-out used a TaskGroup, so one failing subscriber cancelled
     in-flight delivery to its healthy siblings.
  6. Enricher built side-channel lookups outside its try block, so a side
     item missing the join field crashed the link even with on_error set.
"""

import asyncio
import unittest

from io_chains import Chain, Collector, Enricher, LinkMetrics, Processor, Relation
from io_chains._internal.sentinel import END_OF_STREAM


class TestChainPostConstructionWiring(unittest.IsolatedAsyncioTestCase):
    async def test_subscribe_after_construction_receives_output(self):
        results = Collector()
        chain = Chain(links=[Processor(), Processor(processor=lambda x: x * 2)], source=[1, 2])
        chain.subscribe(results)
        await chain()
        self.assertEqual(list(results), [2, 4])

    async def test_subscribers_setter_after_construction_receives_output(self):
        results = Collector()
        chain = Chain(links=[Processor(processor=lambda x: x * 2)], source=[1, 2])
        chain.subscribers = results
        await chain()
        self.assertEqual(list(results), [2, 4])

    async def test_subscribe_with_channel_on_nested_chain(self):
        results = Collector()
        inner = Chain(links=[Processor(processor=lambda x: x + 1)])
        outer = Chain(links=[Processor(), inner], source=[1])
        outer.subscribe(results, channel='c')
        await outer()
        envs = list(results)
        self.assertEqual([(e.channel, e.data) for e in envs], [('c', 2)])

    async def test_subscriber_count_and_items_out_reflect_output_link(self):
        chain = Chain(links=[Processor(), Processor()], source=[1, 2, 3])
        chain.subscribe(Collector())
        await chain()
        self.assertEqual(chain.subscriber_count, 1)
        self.assertEqual(chain.items_out, 3)

    async def test_error_subscribers_reach_every_inner_link(self):
        errors = Collector()
        chain = Chain(
            links=[Processor(processor=lambda x: 1 / 0 if x == 1 else x), Processor(processor=lambda x: 1 / 0)],
            source=[1, 2],
        )
        chain.error_subscribers = errors
        await chain()
        await errors.push(END_OF_STREAM)
        failed = sorted(env.datum for env in errors)
        self.assertEqual(failed, [1, 2])  # 1 failed in link 0, 2 failed in link 1


class TestCancellationDoesNotLeakEos(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_with_full_bounded_queues_leaves_no_pending_tasks(self):
        async def slow(x):
            await asyncio.sleep(10)
            return x

        chain = Chain(
            links=[
                Processor(queue_size=1),
                Processor(processor=slow, queue_size=1),
                Processor(queue_size=1),
            ],
            source=range(100),
            subscribers=Collector(),
        )
        task = asyncio.create_task(chain())
        await asyncio.sleep(0.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.05)
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        self.assertEqual(pending, [], f'leaked tasks: {[t.get_coro() for t in pending]}')

    async def test_cancelled_upstream_still_delivers_eos_to_live_downstream(self):
        async def endless():
            i = 0
            while True:
                yield i
                i += 1
                await asyncio.sleep(0)

        results = Collector(maxsize=1)
        down = Processor(queue_size=1, subscribers=results)
        up = Processor(source=endless, subscribers=down)
        down_task = asyncio.create_task(down())
        up_task = asyncio.create_task(up())

        async def consume():
            return [x async for x in results]

        consumer = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        up_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await up_task
        # Downstream must terminate on its own once the EOS arrives.
        await asyncio.wait_for(down_task, 1)
        await asyncio.wait_for(consumer, 1)


class TestEnricherSameChannelRelations(unittest.IsolatedAsyncioTestCase):
    async def test_two_relations_on_same_channel_with_different_fields(self):
        results = Collector()
        enricher = Enricher(
            primary_channel='p',
            subscribers=results,
            relations=[
                Relation(from_field='a_id', to_channel='side', to_field='id', attach_as='by_id'),
                Relation(from_field='a_code', to_channel='side', to_field='code', attach_as='by_code'),
            ],
        )
        primary = Processor(source=[{'a_id': 1, 'a_code': 'x'}])
        side = Processor(source=[{'id': 1, 'code': 'x'}])
        primary.subscribe(enricher, channel='p')
        side.subscribe(enricher, channel='side')
        await asyncio.gather(primary(), side(), enricher())
        (row,) = list(results)
        self.assertEqual(row['by_id'], {'id': 1, 'code': 'x'})
        self.assertEqual(row['by_code'], {'id': 1, 'code': 'x'})


class TestEnricherSideItemErrors(unittest.IsolatedAsyncioTestCase):
    async def _run(self, on_error):
        results = Collector()
        captured: list[LinkMetrics] = []
        enricher = Enricher(
            primary_channel='p',
            subscribers=results,
            on_error=on_error,
            on_metrics=captured.append,
            relations=[Relation(from_field='loc_id', to_channel='loc', to_field='id', attach_as='loc')],
        )
        primary = Processor(source=[{'loc_id': 1}])
        side = Processor(source=[{'name': 'no id field'}, {'id': 1, 'name': 'Earth'}])
        primary.subscribe(enricher, channel='p')
        side.subscribe(enricher, channel='loc')
        await asyncio.gather(primary(), side(), enricher())
        return list(results), captured[0]

    async def test_side_item_missing_join_field_goes_to_on_error(self):
        seen = []
        rows, metrics = await self._run(lambda e, item: seen.append((type(e), item)))
        self.assertEqual(seen, [(KeyError, {'name': 'no id field'})])
        self.assertEqual(rows, [{'loc_id': 1, 'loc': {'id': 1, 'name': 'Earth'}}])
        self.assertEqual(metrics.items_errored, 1)

    async def test_side_item_missing_join_field_raises_without_on_error(self):
        with self.assertRaises(KeyError):
            await self._run(None)


class TestRetryErrorMetric(unittest.IsolatedAsyncioTestCase):
    async def test_item_that_succeeds_after_retries_is_not_counted_errored(self):
        attempts = {'n': 0}

        def flaky(x):
            attempts['n'] += 1
            if attempts['n'] < 3:
                raise ValueError('transient')
            return x

        results = Collector()
        p = Processor(source=[1], processor=flaky, max_retries=3, subscribers=results)
        await p()
        self.assertEqual(list(results), [1])
        self.assertEqual(p.items_errored, 0)

    async def test_item_that_exhausts_retries_is_counted_once(self):
        p = Processor(
            source=[1], processor=lambda x: 1 / 0, max_retries=3, on_error=lambda e, d: None, subscribers=Collector()
        )
        await p()  # on_error returning None drops the item
        self.assertEqual(p.items_errored, 1)


class _FailingSubscriber(Collector):
    async def push(self, datum):
        if datum == 2:
            raise RuntimeError('bad subscriber')
        await super().push(datum)


class TestFanOutIsolation(unittest.IsolatedAsyncioTestCase):
    async def test_failing_subscriber_does_not_starve_sibling(self):
        healthy = Collector()
        # Failing subscriber listed FIRST so it raises before the sibling runs.
        p = Processor(source=[1, 2, 3], subscribers=[_FailingSubscriber(), healthy])
        with self.assertRaises(RuntimeError):
            await p()
        self.assertEqual(list(healthy), [1, 2])

    async def test_multiple_failing_subscribers_raise_exception_group(self):
        p = Processor(source=[2], subscribers=[_FailingSubscriber(), _FailingSubscriber()])
        with self.assertRaises(ExceptionGroup) as ctx:
            await p()
        self.assertEqual(len(ctx.exception.exceptions), 2)


if __name__ == '__main__':
    unittest.main()
