// SPDX-FileCopyrightText: 2026 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

package opus

import (
	"bytes"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// repacketizerTOC is config 9 (SILK-only WB, 20 ms frames), mono.
const repacketizerTOC = 0b01001000

func TestSplitPacket(t *testing.T) {
	t.Parallel()

	t.Run("code 0 single frame", func(t *testing.T) {
		t.Parallel()

		packet := []byte{repacketizerTOC | byte(frameCodeOneFrame), 10, 20, 30}
		frames, err := SplitPacket(packet)

		require.NoError(t, err)
		assert.Equal(t, [][]byte{{10, 20, 30}}, frames)
	})

	t.Run("code 1 two equal frames", func(t *testing.T) {
		t.Parallel()

		packet := []byte{repacketizerTOC | byte(frameCodeTwoEqualFrames), 1, 2, 3, 4}
		frames, err := SplitPacket(packet)

		require.NoError(t, err)
		assert.Equal(t, [][]byte{{1, 2}, {3, 4}}, frames)
	})

	t.Run("code 2 two different frames", func(t *testing.T) {
		t.Parallel()

		packet := []byte{repacketizerTOC | byte(frameCodeTwoDifferentFrames), 3, 1, 2, 3, 4, 5}
		frames, err := SplitPacket(packet)

		require.NoError(t, err)
		assert.Equal(t, [][]byte{{1, 2, 3}, {4, 5}}, frames)
	})

	t.Run("code 3 VBR multiple frames", func(t *testing.T) {
		t.Parallel()

		packet := []byte{
			repacketizerTOC | byte(frameCodeArbitraryFrames),
			0b10000011, // VBR, no padding, 3 frames
			2, 3,       // lengths of first two frames
			1, 2, 3, 4, 5, 6,
		}
		frames, err := SplitPacket(packet)

		require.NoError(t, err)
		assert.Equal(t, [][]byte{{1, 2}, {3, 4, 5}, {6}}, frames)
	})

	t.Run("code 3 CBR multiple frames", func(t *testing.T) {
		t.Parallel()

		packet := []byte{
			repacketizerTOC | byte(frameCodeArbitraryFrames),
			0b00000011, // CBR, no padding, 3 frames
			1, 2, 3, 4, 5, 6,
		}
		frames, err := SplitPacket(packet)

		require.NoError(t, err)
		assert.Equal(t, [][]byte{{1, 2}, {3, 4}, {5, 6}}, frames)
	})

	t.Run("malformed length", func(t *testing.T) {
		t.Parallel()

		_, err := SplitPacket([]byte{repacketizerTOC | byte(frameCodeTwoDifferentFrames), 5, 1})

		require.Error(t, err)
		assert.ErrorIs(t, err, errMalformedPacket)
	})

	t.Run("empty packet", func(t *testing.T) {
		t.Parallel()

		_, err := SplitPacket(nil)

		require.Error(t, err)
		assert.ErrorIs(t, err, errMalformedPacket)
		assert.ErrorIs(t, err, errTooShortForTableOfContentsHeader)
	})
}

func TestRepacketizerPopFramesPacketBytes(t *testing.T) {
	t.Parallel()

	push := func(r *Repacketizer, frames ...[]byte) {
		t.Helper()
		timestamp := uint32(0)
		for i, frame := range frames {
			packet := append([]byte{repacketizerTOC | byte(frameCodeOneFrame)}, frame...)
			require.NoError(t, r.Push(timestamp+uint32(i)*960, packet))
		}
	}

	t.Run("single frame uses code 0", func(t *testing.T) {
		t.Parallel()

		r := NewRepacketizer()
		push(r, []byte{1, 2, 3})

		packet, err := r.PopFrames(1)
		require.NoError(t, err)
		assert.Equal(t, []byte{repacketizerTOC | byte(frameCodeOneFrame), 1, 2, 3}, packet)
		assert.Zero(t, r.BufferedFrames())
	})

	t.Run("two equal frames use code 1", func(t *testing.T) {
		t.Parallel()

		r := NewRepacketizer()
		push(r, []byte{1, 2}, []byte{3, 4})

		packet, err := r.PopFrames(2)
		require.NoError(t, err)
		assert.Equal(t, []byte{repacketizerTOC | byte(frameCodeTwoEqualFrames), 1, 2, 3, 4}, packet)
	})

	t.Run("two different frames use code 2 with length byte", func(t *testing.T) {
		t.Parallel()

		r := NewRepacketizer()
		push(r, []byte{1, 2, 3}, []byte{4, 5})

		packet, err := r.PopFrames(2)
		require.NoError(t, err)
		assert.Equal(t, []byte{repacketizerTOC | byte(frameCodeTwoDifferentFrames), 3, 1, 2, 3, 4, 5}, packet)
	})

	t.Run("three equal frames use code 3 CBR", func(t *testing.T) {
		t.Parallel()

		r := NewRepacketizer()
		push(r, []byte{1, 2}, []byte{3, 4}, []byte{5, 6})

		packet, err := r.PopFrames(3)
		require.NoError(t, err)
		assert.Equal(t, []byte{
			repacketizerTOC | byte(frameCodeArbitraryFrames),
			0b00000011, // CBR, 3 frames
			1, 2, 3, 4, 5, 6,
		}, packet)
	})

	t.Run("three different frames use code 3 VBR with length bytes", func(t *testing.T) {
		t.Parallel()

		r := NewRepacketizer()
		push(r, []byte{1, 2}, []byte{3, 4, 5}, []byte{6})

		packet, err := r.PopFrames(3)
		require.NoError(t, err)
		assert.Equal(t, []byte{
			repacketizerTOC | byte(frameCodeArbitraryFrames),
			0b10000011, // VBR, 3 frames
			2, 3,       // lengths of the first two frames
			1, 2, 3, 4, 5, 6,
		}, packet)
	})

	t.Run("two-byte frame length", func(t *testing.T) {
		t.Parallel()

		long := bytes.Repeat([]byte{0xAB}, 300)
		r := NewRepacketizer()
		push(r, long, []byte{7})

		packet, err := r.PopFrames(2)
		require.NoError(t, err)
		want := append([]byte{repacketizerTOC | byte(frameCodeTwoDifferentFrames), 252 + (300-252)%4, (300 - 252) / 4}, long...)
		want = append(want, 7)
		assert.Equal(t, want, packet)

		frames, err := SplitPacket(packet)
		require.NoError(t, err)
		assert.Equal(t, [][]byte{long, {7}}, frames)
	})

	t.Run("frames are consumed in order", func(t *testing.T) {
		t.Parallel()

		r := NewRepacketizer()
		push(r, []byte{1}, []byte{2}, []byte{3})

		packet, err := r.PopFrames(2)
		require.NoError(t, err)
		assert.Equal(t, []byte{repacketizerTOC | byte(frameCodeTwoEqualFrames), 1, 2}, packet)
		assert.Equal(t, 1, r.BufferedFrames())

		packet, err = r.PopFrames(1)
		require.NoError(t, err)
		assert.Equal(t, []byte{repacketizerTOC | byte(frameCodeOneFrame), 3}, packet)
	})
}

func TestRepacketizerPopMTU(t *testing.T) {
	t.Parallel()

	pushFrames := func(r *Repacketizer, sizes ...int) {
		t.Helper()
		for i, size := range sizes {
			packet := append([]byte{repacketizerTOC | byte(frameCodeOneFrame)}, bytes.Repeat([]byte{byte(i + 1)}, size)...)
			require.NoError(t, r.Push(uint32(i)*960, packet))
		}
	}

	t.Run("boundary MTU fits exactly", func(t *testing.T) {
		t.Parallel()

		r := NewRepacketizer()
		pushFrames(r, 10, 10, 10)

		// Three equal frames as code 3 CBR: TOC + count byte + 30 payload bytes.
		packet, err := r.PopMTU(32)
		require.NoError(t, err)
		assert.Len(t, packet, 32)
		assert.Zero(t, r.BufferedFrames())
	})

	t.Run("one byte below boundary drops a frame", func(t *testing.T) {
		t.Parallel()

		r := NewRepacketizer()
		pushFrames(r, 10, 10, 10)

		packet, err := r.PopMTU(31)
		require.NoError(t, err)
		// Two equal frames as code 1: TOC + 20 payload bytes.
		assert.Len(t, packet, 21)
		assert.Equal(t, 1, r.BufferedFrames())
	})

	t.Run("MTU between code 1 and code 2 layouts", func(t *testing.T) {
		t.Parallel()

		repacketizer := NewRepacketizer()
		pushFrames(repacketizer, 10, 20)

		// Code 2 packet: TOC + length byte + 30 payload bytes = 32.
		packet, err := repacketizer.PopMTU(32)
		require.NoError(t, err)
		assert.Len(t, packet, 32)

		repacketizer = NewRepacketizer()
		pushFrames(repacketizer, 10, 20)

		packet, err = repacketizer.PopMTU(31)
		require.NoError(t, err)
		// Only the first frame fits, as code 0: TOC + 10 payload bytes.
		assert.Len(t, packet, 11)
		assert.Equal(t, 1, repacketizer.BufferedFrames())
	})

	t.Run("single frame exceeding MTU is an error", func(t *testing.T) {
		t.Parallel()

		r := NewRepacketizer()
		pushFrames(r, 100)

		_, err := r.PopMTU(50)
		require.ErrorIs(t, err, errRepacketizerFrameExceedsMTU)
		assert.Equal(t, 1, r.BufferedFrames())
	})

	t.Run("invalid MTU", func(t *testing.T) {
		t.Parallel()

		r := NewRepacketizer()
		pushFrames(r, 10)

		_, err := r.PopMTU(0)
		require.ErrorIs(t, err, errInvalidRepacketizeMTU)
	})

	t.Run("empty buffer", func(t *testing.T) {
		t.Parallel()

		_, err := NewRepacketizer().PopMTU(1200)
		require.ErrorIs(t, err, errRepacketizerNotEnoughFrames)
	})

	t.Run("duration cap limits frames per packet", func(t *testing.T) {
		t.Parallel()

		r := NewRepacketizer()
		// 20 ms frames: at most 6 (120 ms) per packet even with a huge MTU.
		pushFrames(r, 1, 1, 1, 1, 1, 1, 1, 1)

		packet, err := r.PopMTU(1200)
		require.NoError(t, err)
		frames, err := SplitPacket(packet)
		require.NoError(t, err)
		assert.Len(t, frames, 6)
		assert.Equal(t, 2, r.BufferedFrames())
	})
}

func TestRepacketizerErrors(t *testing.T) {
	t.Parallel()

	code0Packet := func(toc byte, payload ...byte) []byte {
		return append([]byte{toc | byte(frameCodeOneFrame)}, payload...)
	}

	t.Run("TOC configuration mismatch", func(t *testing.T) {
		t.Parallel()

		r := NewRepacketizer()
		require.NoError(t, r.Push(0, code0Packet(repacketizerTOC, 1)))

		err := r.Push(960, code0Packet(0b01010000, 1)) // config 10, 40 ms frames
		require.ErrorIs(t, err, errRepacketizerConfigMismatch)
	})

	t.Run("stereo flag mismatch", func(t *testing.T) {
		t.Parallel()

		r := NewRepacketizer()
		require.NoError(t, r.Push(0, code0Packet(repacketizerTOC, 1)))

		err := r.Push(960, code0Packet(repacketizerTOC|0b100, 1))
		require.ErrorIs(t, err, errRepacketizerConfigMismatch)
	})

	t.Run("malformed packet length", func(t *testing.T) {
		t.Parallel()

		r := NewRepacketizer()
		err := r.Push(0, []byte{repacketizerTOC | byte(frameCodeTwoDifferentFrames), 9, 1})
		require.ErrorIs(t, err, errMalformedPacket)
		assert.Zero(t, r.BufferedFrames())
	})

	t.Run("oversized frame", func(t *testing.T) {
		t.Parallel()

		r := NewRepacketizer()
		err := r.Push(0, code0Packet(repacketizerTOC, make([]byte, maxOpusFrameSize+1)...))
		require.ErrorIs(t, err, errMalformedPacket)
	})

	t.Run("timestamp gap", func(t *testing.T) {
		t.Parallel()

		r := NewRepacketizer()
		require.NoError(t, r.Push(1000, code0Packet(repacketizerTOC, 1)))

		err := r.Push(1000, code0Packet(repacketizerTOC, 2)) // want 1000 + 960
		require.ErrorIs(t, err, errRepacketizerTimestampGap)
		assert.Equal(t, 1, r.BufferedFrames())

		// The expected timestamp is accepted after the rejected packet.
		require.NoError(t, r.Push(1960, code0Packet(repacketizerTOC, 2)))
		assert.Equal(t, 2, r.BufferedFrames())
	})

	t.Run("timestamp gap across multi-frame packets", func(t *testing.T) {
		t.Parallel()

		r := NewRepacketizer()
		// Code 1 packet: two 20 ms frames, 1920 samples.
		require.NoError(t, r.Push(0, []byte{repacketizerTOC | byte(frameCodeTwoEqualFrames), 1, 2}))

		err := r.Push(960, code0Packet(repacketizerTOC, 3))
		require.ErrorIs(t, err, errRepacketizerTimestampGap)

		require.NoError(t, r.Push(1920, code0Packet(repacketizerTOC, 3)))
	})

	t.Run("missing frames", func(t *testing.T) {
		t.Parallel()

		r := NewRepacketizer()
		require.NoError(t, r.Push(0, code0Packet(repacketizerTOC, 1)))

		_, err := r.PopFrames(2)
		require.ErrorIs(t, err, errRepacketizerNotEnoughFrames)
		assert.Equal(t, 1, r.BufferedFrames())
	})

	t.Run("invalid target frame count", func(t *testing.T) {
		t.Parallel()

		r := NewRepacketizer()
		require.NoError(t, r.Push(0, code0Packet(repacketizerTOC, 1)))

		for _, count := range []int{-1, 0, 49} {
			_, err := r.PopFrames(count)
			require.ErrorIs(t, err, errInvalidRepacketizeFrameCount, "count %d", count)
		}
	})

	t.Run("target frame count exceeds 120 ms", func(t *testing.T) {
		t.Parallel()

		r := NewRepacketizer()
		require.NoError(t, r.Push(0, code0Packet(repacketizerTOC, 1)))

		// 20 ms frames: 7 frames would be 140 ms.
		_, err := r.PopFrames(7)
		require.ErrorIs(t, err, errInvalidRepacketizeFrameCount)
	})
}

// TestRepacketizerPCMEquivalence encodes a signal into single-frame
// packets, repacketizes them into multi-frame packets, and verifies that
// decoding the repacketized stream yields PCM identical to decoding the
// original packets.
func TestRepacketizerPCMEquivalence(t *testing.T) {
	t.Parallel()

	encoder, err := NewEncoder(WithVBR(true))
	require.NoError(t, err)

	const frameSamples = 960 // 20 ms at 48 kHz
	const packetCount = 12

	pcm := make([]float32, frameSamples)
	packets := make([][]byte, 0, packetCount)
	for i := range packetCount {
		for j := range pcm {
			// Varying amplitude produces VBR frames of differing sizes.
			pcm[j] = float32(math.Sin(2*math.Pi*440*float64(j)/48000)) * float32(i%3+1) / 6
		}
		packet := make([]byte, maxOpusFrameSize)
		n, err := encoder.EncodeFloat32(pcm, packet)
		require.NoError(t, err)
		packets = append(packets, packet[:n])
	}

	// Repacketize: pairs for the first half, triples for the second.
	repacketizer := NewRepacketizer()
	for i, packet := range packets {
		require.NoError(t, repacketizer.Push(uint32(i*frameSamples), packet))
	}
	var repacketized [][]byte
	for range packetCount / 2 / 2 {
		packet, err := repacketizer.PopFrames(2)
		require.NoError(t, err)
		repacketized = append(repacketized, packet)
	}
	for range packetCount / 2 / 3 {
		packet, err := repacketizer.PopFrames(3)
		require.NoError(t, err)
		repacketized = append(repacketized, packet)
	}
	assert.Zero(t, repacketizer.BufferedFrames())

	// Every coded frame survives, byte-identical and in order.
	var wantFrames, gotFrames [][]byte
	for _, packet := range packets {
		frames, err := SplitPacket(packet)
		require.NoError(t, err)
		wantFrames = append(wantFrames, frames...)
	}
	for _, packet := range repacketized {
		frames, err := SplitPacket(packet)
		require.NoError(t, err)
		gotFrames = append(gotFrames, frames...)
	}
	require.Equal(t, wantFrames, gotFrames)

	decodeAll := func(stream [][]byte) []float32 {
		t.Helper()
		decoder := NewDecoder()
		var out []float32
		buf := make([]float32, 6*frameSamples)
		for _, packet := range stream {
			n, err := decoder.DecodeToFloat32(packet, buf)
			require.NoError(t, err)
			out = append(out, buf[:n]...)
		}

		return out
	}

	assert.Equal(t, decodeAll(packets), decodeAll(repacketized))
}

// TestRepacketizerSplitRoundTrip splits multi-frame packets and rebuilds
// them, checking the frames and decoded PCM survive unchanged.
func TestRepacketizerSplitRoundTrip(t *testing.T) {
	t.Parallel()

	// A code 3 VBR packet with three 20 ms frames.
	original := []byte{
		repacketizerTOC | byte(frameCodeArbitraryFrames),
		0b10000011,
		2, 3,
		1, 2, 3, 4, 5, 6,
	}

	frames, err := SplitPacket(original)
	require.NoError(t, err)
	require.Len(t, frames, 3)

	r := NewRepacketizer()
	for i, frame := range frames {
		packet := append([]byte{repacketizerTOC | byte(frameCodeOneFrame)}, frame...)
		require.NoError(t, r.Push(uint32(i)*960, packet))
	}

	rebuilt, err := r.PopFrames(3)
	require.NoError(t, err)
	assert.Equal(t, original, rebuilt)
}
