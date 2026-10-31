"""Sliding-window logic (no sockets): SendWindow and ReceiveBuffer."""

import unittest

from protocol.window import ReceiveBuffer, SendWindow


def fill(win, now=0.0, rto=1.0):
    while win.can_send_new():
        win.on_new_send(now, now + rto, 0)


class TestSendWindow(unittest.TestCase):
    def test_window_limits_packets_in_flight(self):
        win = SendWindow(total_chunks=10, window=4)
        fill(win)
        self.assertEqual(win.next_seq, 4)
        self.assertFalse(win.can_send_new())
        self.assertEqual(sorted(win.inflight), [0, 1, 2, 3])

    def test_window_slides_on_in_order_ack(self):
        win = SendWindow(10, 4)
        fill(win)
        sample, newly = win.on_ack(0, 1, now=0.1)
        self.assertAlmostEqual(sample, 0.1)
        self.assertEqual((newly, win.base), (1, 1))
        self.assertTrue(win.can_send_new())

    def test_selective_ack_out_of_order_does_not_slide(self):
        win = SendWindow(10, 4)
        fill(win)
        win.on_ack(2, 0, 0.1)          # packet 2 arrived, 0 still missing
        win.on_ack(1, 0, 0.1)
        self.assertEqual(win.base, 0)
        self.assertFalse(win.can_send_new())
        win.on_ack(0, 3, 0.2)          # the hole is filled: base jumps over 1 and 2
        self.assertEqual(win.base, 3)

    def test_cumulative_ack_covers_lost_acks(self):
        win = SendWindow(10, 4)
        fill(win)
        sample, newly = win.on_ack(3, 4, 0.1)   # ACKs for 0,1,2 were lost
        self.assertEqual((newly, win.base), (4, 4))
        self.assertEqual(win.inflight, {})

    def test_duplicate_ack_ignored(self):
        win = SendWindow(10, 4)
        fill(win)
        win.on_ack(0, 1, 0.1)
        sample, newly = win.on_ack(0, 1, 0.2)
        self.assertIsNone(sample)
        self.assertEqual(newly, 0)

    def test_karn_no_sample_from_retransmitted_packet(self):
        win = SendWindow(10, 4)
        fill(win)
        win.on_retransmit(0, now=1.0, deadline=3.0, epoch=1)
        sample, newly = win.on_ack(0, 1, now=1.05)
        self.assertIsNone(sample)      # ambiguous: original or retransmission?
        self.assertEqual(newly, 1)
        sample, _ = win.on_ack(1, 2, now=1.06)
        self.assertAlmostEqual(sample, 1.06)   # never retransmitted: valid

    def test_ack_for_unsent_data_is_clamped(self):
        win = SendWindow(10, 4)
        fill(win)
        sample, newly = win.on_ack(9, 10, 0.1)  # bogus: 9 was never sent
        self.assertEqual(win.base, 4)           # only the 4 sent packets count
        self.assertEqual(newly, 4)
        self.assertIsNone(sample)

    def test_timers_expire_in_order_and_skip_acked(self):
        win = SendWindow(10, 3)
        for i, dl in enumerate([0.5, 0.3, 0.4]):
            win.on_new_send(0.0, dl, 0)
        self.assertAlmostEqual(win.next_deadline(), 0.3)
        win.on_ack(1, 0, 0.1)                   # packet 1 acked: its timer is stale
        self.assertAlmostEqual(win.next_deadline(), 0.4)
        expired = win.pop_expired(0.45)
        self.assertEqual([e.seq for e in expired], [2])
        self.assertEqual([e.seq for e in win.pop_expired(1.0)], [0])

    def test_retransmit_rearms_timer(self):
        win = SendWindow(10, 2)
        fill(win)
        self.assertEqual([x.seq for x in win.pop_expired(1.0)], [0, 1])
        win.on_retransmit(0, 1.0, 2.5, epoch=1)
        self.assertEqual(win.inflight[0].retx, 1)
        self.assertEqual([x.seq for x in win.pop_expired(2.0)], [])
        self.assertEqual([x.seq for x in win.pop_expired(2.6)], [0])

    def test_rearm_all(self):
        win = SendWindow(10, 3)
        fill(win, now=0.0, rto=5.0)
        win.rearm_all(rto=0.2, epoch=7)
        self.assertEqual(sorted(e.seq for e in win.pop_expired(0.25)), [0, 1, 2])

    def test_done(self):
        win = SendWindow(2, 4)
        fill(win)
        self.assertFalse(win.done())
        win.on_ack(1, 2, 0.1)
        self.assertTrue(win.done())
        self.assertTrue(SendWindow(0, 4).done())   # empty file


class TestRack(unittest.TestCase):
    """RACK (RFC 8985) loss detection, with explicit times."""

    def test_disabled_by_default(self):
        win = SendWindow(10, 8)
        fill(win)
        win.on_ack(1, 0, 0.1)
        self.assertEqual(win.detect_losses(5.0), [])
        self.assertIsNone(win.next_rack_deadline())

    def test_loss_declared_after_rtt_plus_reordering_window(self):
        win = SendWindow(10, 8, rack=True)
        win.on_new_send(0.000, 9, 0)        # 0 (will be lost)
        win.on_new_send(0.001, 9, 0)        # 1
        win.on_ack(1, 0, 0.101)             # 1 delivered: RTT 0.100, sent after 0
        # 0 is lost once outstanding > RTT + min_rtt/4 = 0.100 + 0.025
        self.assertAlmostEqual(win.next_rack_deadline(), 0.125)
        self.assertEqual(win.detect_losses(0.120), [])            # still within the window
        self.assertEqual([e.seq for e in win.detect_losses(0.126)], [0])
        self.assertEqual(win.detect_losses(0.2), [])              # reported once only

    def test_mild_reordering_is_not_loss(self):
        win = SendWindow(10, 8, rack=True)
        win.on_new_send(0.000, 9, 0)
        win.on_new_send(0.001, 9, 0)
        win.on_ack(1, 0, 0.100)             # 1 overtakes 0 ...
        win.on_ack(0, 2, 0.105)             # ... but 0 arrives 5 ms later
        self.assertEqual(win.detect_losses(1.0), [])

    def test_packets_sent_after_the_delivered_one_are_not_judged(self):
        win = SendWindow(10, 8, rack=True)
        win.on_new_send(0.0, 9, 0)          # 0
        win.on_new_send(0.5, 9, 0)          # 1 sent later than 0
        win.on_ack(0, 1, 0.1)
        self.assertEqual(win.detect_losses(10.0), [])             # nothing newer was delivered

    def test_resent_packet_judged_against_its_new_send_time(self):
        win = SendWindow(10, 8, rack=True)
        win.on_new_send(0.000, 9, 0)
        win.on_new_send(0.001, 9, 0)
        win.on_ack(1, 0, 0.101)
        self.assertEqual([e.seq for e in win.detect_losses(0.2)], [0])
        win.on_retransmit(0, 0.2, 9, 0)
        self.assertEqual(win.detect_losses(0.3), [])              # resent after the newest delivery
        win.on_new_send(0.21, 9, 0)                               # 2, sent after the resend
        win.on_ack(2, 0, 0.31)
        self.assertEqual([e.seq for e in win.detect_losses(0.34)], [0])

    def test_ignores_too_fast_ack_for_resent_packet(self):
        win = SendWindow(10, 8, rack=True)
        win.on_new_send(0.0, 9, 0)
        win.on_new_send(0.0, 9, 0)
        win.on_ack(1, 0, 0.1)               # min RTT 0.1
        win.on_retransmit(0, 0.5, 9, 0)
        win.on_ack(0, 2, 0.52)              # 20 ms after resend: ACK of the original copy
        self.assertAlmostEqual(win.rack_rtt, 0.1)


class TestRackAdaptation(unittest.TestCase):
    def test_spurious_resend_widens_reordering_window(self):
        win = SendWindow(10, 8, rack=True)
        win.on_new_send(0.000, 9, 0)
        win.on_new_send(0.001, 9, 0)
        win.on_ack(1, 0, 0.101)                                  # min RTT 0.1
        lost = win.detect_losses(0.13)
        self.assertEqual([e.seq for e in lost], [0])
        win.on_retransmit(0, 0.13, 9, 0, by_rack=True)
        win.on_ack(0, 2, 0.135)                                  # 5 ms later: the original copy
        self.assertEqual(win.spurious_detected, 1)
        self.assertEqual(win.reo_mult, 2)
        # next time a packet waits RTT + 2 x 25 ms before being declared lost
        win.on_new_send(0.20, 9, 0)                              # 2
        win.on_new_send(0.201, 9, 0)                             # 3
        win.on_ack(3, 2, 0.301)
        self.assertAlmostEqual(win.next_rack_deadline(), 0.20 + 0.1 + 0.05)

    def test_window_never_exceeds_one_rtt(self):
        win = SendWindow(10, 8, rack=True)
        win.on_new_send(0.0, 9, 0)
        win.on_new_send(0.001, 9, 0)
        win.on_ack(1, 0, 0.101)
        win.reo_mult = 50
        self.assertAlmostEqual(win.next_rack_deadline(), 0.0 + 0.1 + 0.1)


class TestReceiveBuffer(unittest.TestCase):
    def test_in_order(self):
        b = ReceiveBuffer(3, 4)
        self.assertEqual(b.offer(0, b"a"), (ReceiveBuffer.IN_ORDER, [b"a"]))
        self.assertEqual(b.expected, 1)

    def test_out_of_order_is_buffered_then_released(self):
        b = ReceiveBuffer(4, 4)
        self.assertEqual(b.offer(2, b"c"), (ReceiveBuffer.BUFFERED, []))
        self.assertEqual(b.offer(1, b"b"), (ReceiveBuffer.BUFFERED, []))
        self.assertEqual(b.offer(0, b"a"), (ReceiveBuffer.IN_ORDER, [b"a", b"b", b"c"]))
        self.assertEqual(b.expected, 3)
        self.assertEqual(b.buffer, {})

    def test_duplicates(self):
        b = ReceiveBuffer(4, 4)
        b.offer(0, b"a")
        b.offer(2, b"c")
        self.assertEqual(b.offer(0, b"a")[0], ReceiveBuffer.DUPLICATE)   # already written
        self.assertEqual(b.offer(2, b"c")[0], ReceiveBuffer.DUPLICATE)   # already buffered

    def test_out_of_window_dropped_and_memory_bounded(self):
        b = ReceiveBuffer(1000, 4)
        self.assertEqual(b.offer(4, b"x")[0], ReceiveBuffer.OUT_OF_WINDOW)
        for s in range(1, 1000):
            b.offer(s, b"x")
        self.assertLessEqual(len(b.buffer), 4)

    def test_invalid_seq(self):
        self.assertEqual(ReceiveBuffer(3, 4).offer(3, b"x")[0], ReceiveBuffer.INVALID)

    def test_complete(self):
        b = ReceiveBuffer(2, 4)
        self.assertFalse(b.complete())
        b.offer(1, b"b")
        b.offer(0, b"a")
        self.assertTrue(b.complete())
        self.assertTrue(ReceiveBuffer(0, 4).complete())


if __name__ == "__main__":
    unittest.main()
