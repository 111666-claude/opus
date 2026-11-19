// SPDX-FileCopyrightText: 2026 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

package opus

import "fmt"

// SplitPacket parses an Opus packet into its coded frames, preserving each
// frame's bytes and their order. It supports all frame count codes (0-3),
// CBR and VBR, and padding, per RFC 6716 Section 3.2. The returned frames
// alias packet; copy them if the packet buffer is reused.
func SplitPacket(packet []byte) ([][]byte, error) {
	if len(packet) < 1 {
		return nil, fmt.Errorf("%w: %w", errMalformedPacket, errTooShortForTableOfContentsHeader)
	}

	return parsePacketFrames(packet, tableOfContentsHeader(packet[0]))
}

// Repacketizer recombines consecutive Opus packets of a single stream into
// new packets with a different frame layout, without decoding or
// re-encoding the audio. A forwarder can use it to merge small packets and
// reduce per-packet overhead: push received packets with their RTP
// timestamps, then pop packets built to a target frame count or MTU.
//
// All pushed packets must share the same TOC configuration and stereo flag
// and carry contiguous RTP timestamps (48 kHz clock); violations are
// reported as errors and leave the buffered state untouched. The zero value
// is ready to use.
type Repacketizer struct {
	toc           tableOfContentsHeader // configuration and stereo bits; frame code is always cleared
	frames        [][]byte
	nextTimestamp uint32
	started       bool
}

// NewRepacketizer returns a Repacketizer ready to accept packets.
func NewRepacketizer() *Repacketizer {
	return &Repacketizer{}
}

// Reset drops all buffered frames and stream state, allowing the
// Repacketizer to be reused for a new stream or after a discontinuity.
func (r *Repacketizer) Reset() {
	*r = Repacketizer{}
}

// Push parses packet and buffers its coded frames in order. timestamp is
// the packet's RTP timestamp on the Opus 48 kHz clock. Packets must arrive
// in order with no gaps: the expected timestamp of each packet is the
// previous packet's timestamp plus its decoded duration in samples.
//
// Push reports errMalformedPacket for malformed packets,
// errRepacketizerConfigMismatch when the TOC configuration or stereo flag
// differs from previous packets, and errRepacketizerTimestampGap for
// non-contiguous timestamps. On error no frames are buffered.
func (r *Repacketizer) Push(timestamp uint32, packet []byte) error {
	frames, err := SplitPacket(packet)
	if err != nil {
		return err
	}

	toc := tableOfContentsHeader(packet[0]) & 0b11111100
	if r.started && toc != r.toc {
		return fmt.Errorf("%w: got 0x%02x, stream uses 0x%02x", errRepacketizerConfigMismatch, byte(toc), byte(r.toc))
	}

	frameDuration := toc.configuration().frameDuration()
	if r.started && timestamp != r.nextTimestamp {
		return fmt.Errorf("%w: got %d, want %d", errRepacketizerTimestampGap, timestamp, r.nextTimestamp)
	}

	for _, frame := range frames {
		r.frames = append(r.frames, append([]byte(nil), frame...))
	}
	r.toc = toc
	r.started = true
	r.nextTimestamp = timestamp + uint32(len(frames))*uint32(frameDuration.samples48k()) //nolint:gosec // G115: bounded by 120 ms per packet.

	return nil
}

// BufferedFrames returns the number of coded frames currently buffered.
func (r *Repacketizer) BufferedFrames() int {
	return len(r.frames)
}

// PopFrames builds a single Opus packet containing exactly frameCount of
// the buffered frames, in order, and removes them from the buffer. The
// frame count code (0-3) and CBR/VBR layout are chosen from the frame
// sizes, per RFC 6716 Section 3.2.
//
// frameCount must be between 1 and 48 and the resulting packet duration
// must not exceed 120 ms, otherwise errInvalidRepacketizeFrameCount is
// returned. If fewer than frameCount frames are buffered,
// errRepacketizerNotEnoughFrames is returned and the buffer is unchanged.
func (r *Repacketizer) PopFrames(frameCount int) ([]byte, error) {
	if err := r.checkFrameCount(frameCount); err != nil {
		return nil, err
	}
	if len(r.frames) < frameCount {
		return nil, fmt.Errorf("%w: have %d, want %d", errRepacketizerNotEnoughFrames, len(r.frames), frameCount)
	}

	return r.pop(frameCount), nil
}

// PopMTU builds the largest Opus packet, in whole buffered frames and in
// order, whose total size does not exceed mtu bytes, and removes those
// frames from the buffer. The packet never holds more than 48 frames or
// more than 120 ms of audio.
//
// mtu must be at least 1. If even a single-frame packet does not fit,
// errRepacketizerFrameExceedsMTU is returned. If no frames are buffered,
// errRepacketizerNotEnoughFrames is returned. On error the buffer is
// unchanged.
func (r *Repacketizer) PopMTU(mtu int) ([]byte, error) {
	if mtu < 1 {
		return nil, fmt.Errorf("%w: %d", errInvalidRepacketizeMTU, mtu)
	}
	if len(r.frames) == 0 {
		return nil, fmt.Errorf("%w: have 0, want 1", errRepacketizerNotEnoughFrames)
	}

	frameCount := 0
	for count := 1; count <= r.maxFramesPerPacket() && count <= len(r.frames); count++ {
		if repacketizedSize(r.frames[:count]) > mtu {
			break
		}
		frameCount = count
	}
	if frameCount == 0 {
		return nil, fmt.Errorf("%w: frame of %d bytes does not fit MTU %d",
			errRepacketizerFrameExceedsMTU, repacketizedSize(r.frames[:1]), mtu)
	}

	return r.pop(frameCount), nil
}

// checkFrameCount validates a target frame count against the RFC 6716
// Section 3.2.5 limits: 1-48 frames and at most 120 ms of audio per packet.
func (r *Repacketizer) checkFrameCount(frameCount int) error {
	if frameCount < 1 || frameCount > maxOpusFramesPerPacket {
		return fmt.Errorf("%w: %d not in 1..%d", errInvalidRepacketizeFrameCount, frameCount, maxOpusFramesPerPacket)
	}
	if frameCount*r.toc.configuration().frameDuration().nanoseconds() > maxOpusPacketDurationNanosecond {
		return fmt.Errorf("%w: %d frames of %s exceed 120 ms",
			errInvalidRepacketizeFrameCount, frameCount, r.toc.configuration().frameDuration())
	}

	return nil
}

// maxFramesPerPacket caps a packet at 48 frames and 120 ms of audio.
func (r *Repacketizer) maxFramesPerPacket() int {
	frameCount := maxOpusFramesPerPacket
	if durationCap := maxOpusPacketDurationNanosecond / r.toc.configuration().frameDuration().nanoseconds(); frameCount > durationCap {
		frameCount = durationCap
	}

	return frameCount
}

// pop builds a packet from the first frameCount buffered frames and removes
// them from the buffer.
func (r *Repacketizer) pop(frameCount int) []byte {
	packet := appendRepacketized(nil, r.toc, r.frames[:frameCount])
	copy(r.frames, r.frames[frameCount:])
	clear(r.frames[len(r.frames)-frameCount:])
	r.frames = r.frames[:len(r.frames)-frameCount]

	return packet
}

// frameLengthSize returns the number of bytes needed to signal a frame
// length, per RFC 6716 Section 3.2.1: one byte below 252, two bytes up to
// the 1275-byte maximum.
func frameLengthSize(frameLength int) int {
	if frameLength < 252 {
		return 1
	}

	return 2
}

// appendFrameLength appends the one- or two-byte encoding of frameLength.
func appendFrameLength(dst []byte, frameLength int) []byte {
	if frameLength < 252 {
		return append(dst, byte(frameLength)) //nolint:gosec // G115: frameLength < 252 here.
	}

	return append(dst, byte(252+(frameLength-252)%4), byte((frameLength-252)/4)) //nolint:gosec // G115: frameLength <= 1275.
}

// framesAllEqualSize reports whether every frame has the same length.
func framesAllEqualSize(frames [][]byte) bool {
	for _, frame := range frames[1:] {
		if len(frame) != len(frames[0]) {
			return false
		}
	}

	return true
}

// repacketizedSize returns the size of the packet appendRepacketized would
// build from frames.
func repacketizedSize(frames [][]byte) int {
	size := 1 // TOC byte
	for _, frame := range frames {
		size += len(frame)
	}

	switch {
	case len(frames) == 1:
	case len(frames) == 2:
		if len(frames[0]) != len(frames[1]) {
			size += frameLengthSize(len(frames[0]))
		}
	default:
		size++ // frame count byte
		if !framesAllEqualSize(frames) {
			for _, frame := range frames[:len(frames)-1] {
				size += frameLengthSize(len(frame))
			}
		}
	}

	return size
}

// appendRepacketized appends an Opus packet built from frames, choosing the
// most compact frame count code per RFC 6716 Section 3.2: code 0 for one
// frame, code 1 for two equal-sized frames, code 2 for two differently
// sized frames, and code 3 (CBR when all frames share a size, VBR
// otherwise) for three or more. No padding is emitted. toc carries the
// configuration and stereo bits; its frame code bits are replaced.
func appendRepacketized(dst []byte, toc tableOfContentsHeader, frames [][]byte) []byte {
	switch {
	case len(frames) == 1:
		dst = append(dst, byte(toc)|byte(frameCodeOneFrame))

		return append(dst, frames[0]...)
	case len(frames) == 2 && len(frames[0]) == len(frames[1]):
		dst = append(dst, byte(toc)|byte(frameCodeTwoEqualFrames))
		dst = append(dst, frames[0]...)

		return append(dst, frames[1]...)
	case len(frames) == 2:
		dst = append(dst, byte(toc)|byte(frameCodeTwoDifferentFrames))
		dst = appendFrameLength(dst, len(frames[0]))
		dst = append(dst, frames[0]...)

		return append(dst, frames[1]...)
	default:
		vbr := !framesAllEqualSize(frames)
		frameCountByte := byte(len(frames)) //nolint:gosec // G115: len(frames) <= 48.
		if vbr {
			frameCountByte |= 0b10000000
		}
		dst = append(dst, byte(toc)|byte(frameCodeArbitraryFrames), frameCountByte)
		if vbr {
			for _, frame := range frames[:len(frames)-1] {
				dst = appendFrameLength(dst, len(frame))
			}
		}
		for _, frame := range frames {
			dst = append(dst, frame...)
		}

		return dst
	}
}
