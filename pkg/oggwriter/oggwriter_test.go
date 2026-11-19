// SPDX-FileCopyrightText: 2026 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

package oggwriter

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"strings"
	"testing"

	"github.com/pion/opus"
	"github.com/pion/opus/pkg/oggreader"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type rawPage struct {
	headerType byte
	granule    uint64
	serial     uint32
	sequence   uint32
	lacing     []byte
	payload    []byte
}

type failingWriter struct {
	limit   int
	written int
	err     error
}

func (w *failingWriter) Write(p []byte) (int, error) {
	remaining := w.limit - w.written
	if remaining <= 0 {
		return 0, w.err
	}
	if len(p) <= remaining {
		w.written += len(p)

		return len(p), nil
	}
	w.written += remaining

	return remaining, w.err
}

// parsePages splits a finished Ogg stream into pages without validating it,
// so tests can inspect flags, granule positions and sequence numbers.
func parsePages(t *testing.T, data []byte) []rawPage {
	t.Helper()

	pages := []rawPage{}
	for offset := 0; offset < len(data); {
		require.GreaterOrEqual(t, len(data)-offset, pageHeaderSize, "truncated page header")
		header := data[offset : offset+pageHeaderSize]
		require.Equal(t, pageHeaderSignature, string(header[:4]))
		require.Equal(t, byte(0), header[4])

		segmentCount := int(header[pageSegmentCountOffset])
		require.GreaterOrEqual(t, len(data)-(offset+pageHeaderSize), segmentCount, "truncated lacing table")
		lacing := data[offset+pageHeaderSize : offset+pageHeaderSize+segmentCount]

		payloadSize := 0
		for _, value := range lacing {
			payloadSize += int(value)
		}

		payloadStart := offset + pageHeaderSize + segmentCount
		require.GreaterOrEqual(t, len(data)-payloadStart, payloadSize, "truncated page payload")

		pages = append(pages, rawPage{
			headerType: header[5],
			granule:    binary.LittleEndian.Uint64(header[6:14]),
			serial:     binary.LittleEndian.Uint32(header[14:18]),
			sequence:   binary.LittleEndian.Uint32(header[18:22]),
			lacing:     append([]byte(nil), lacing...),
			payload:    append([]byte(nil), data[payloadStart:payloadStart+payloadSize]...),
		})

		offset = payloadStart + payloadSize
	}

	return pages
}

func readPackets(t *testing.T, data []byte) ([][]byte, *oggreader.OggHeader) {
	t.Helper()

	reader, header, err := oggreader.NewWith(bytes.NewReader(data))
	require.NoError(t, err)

	packets := [][]byte{}
	for {
		packet, _, err := reader.ParseNextPacket()
		if errors.Is(err, io.EOF) {
			break
		}
		require.NoError(t, err)

		packets = append(packets, packet)
	}

	return packets, header
}

func TestNewWithWritesHeaders(t *testing.T) {
	var buffer bytes.Buffer

	writer, err := NewWith(&buffer, 2, WithPreSkip(312), WithVendor("game-assets"), WithSerial(0x1234))
	require.NoError(t, err)
	require.NoError(t, writer.Close())

	reader, header, err := oggreader.NewWith(bytes.NewReader(buffer.Bytes()))
	require.NoError(t, err)
	assert.Equal(t, uint8(2), header.Channels)
	assert.Equal(t, uint16(312), header.PreSkip)
	assert.Equal(t, uint32(48000), header.SampleRate)
	assert.Equal(t, uint8(1), header.Version)
	assert.Equal(t, uint8(0), header.ChannelMap)
	assert.Nil(t, reader.ChannelMapping().Mapping)

	// oggreader consumes the OpusHead packet while parsing the header, so
	// the first packet it returns is OpusTags.
	packets, _ := readPackets(t, buffer.Bytes())
	require.Len(t, packets, 1)
	assert.Equal(t, []byte("OpusTags"), packets[0][:8])

	vendorLength := binary.LittleEndian.Uint32(packets[0][8:12])
	assert.Equal(t, uint32(len("game-assets")), vendorLength)
	assert.Equal(t, "game-assets", string(packets[0][12:12+vendorLength]))
	assert.Equal(t, uint32(0), binary.LittleEndian.Uint32(packets[0][12+vendorLength:])) // no user comments

	pages := parsePages(t, buffer.Bytes())
	require.Len(t, pages, 3)
	assert.Equal(t, byte(pageHeaderTypeBeginningOfStream), pages[0].headerType)
	assert.Equal(t, uint64(0), pages[0].granule)
	assert.Equal(t, byte(0), pages[1].headerType)
	assert.Equal(t, uint64(0), pages[1].granule)
	assert.Equal(t, byte(pageHeaderTypeEndOfStream), pages[2].headerType)
	assert.Equal(t, uint64(0), pages[2].granule)
	for i, page := range pages {
		assert.Equal(t, uint32(0x1234), page.serial)
		assert.Equal(t, uint32(i), page.sequence) //nolint:gosec // small loop index.
	}
}

func TestNewWithDefaults(t *testing.T) {
	var buffer bytes.Buffer

	writer, err := NewWith(&buffer, 1)
	require.NoError(t, err)
	require.NoError(t, writer.Close())

	packets, header := readPackets(t, buffer.Bytes())
	assert.Equal(t, uint8(1), header.Channels)
	assert.Equal(t, uint16(0), header.PreSkip)
	require.Len(t, packets, 1)

	vendorLength := binary.LittleEndian.Uint32(packets[0][8:12])
	assert.Equal(t, defaultVendor, string(packets[0][12:12+vendorLength]))

	pages := parsePages(t, buffer.Bytes())
	for _, page := range pages {
		assert.Equal(t, uint32(defaultSerial), page.serial)
	}
}

func TestNewWithErrors(t *testing.T) {
	var buffer bytes.Buffer

	_, err := NewWith(nil, 1)
	assert.ErrorIs(t, err, errNilWriter)

	_, err = NewWith(&buffer, 0)
	assert.ErrorIs(t, err, errInvalidChannelCount)

	_, err = NewWith(&buffer, 3)
	assert.ErrorIs(t, err, errInvalidChannelCount)

	_, err = NewWith(&buffer, 2, WithVendor("bad\xffutf8"))
	assert.ErrorIs(t, err, errInvalidVendor)

	_, err = NewWith(&buffer, 2, WithVendor(strings.Repeat("v", maxVendorLength+1)))
	assert.ErrorIs(t, err, errVendorTooLong)
	assert.Zero(t, buffer.Len(), "a rejected option must not write a partial stream")

	_, err = NewWith(&failingWriter{limit: 0, err: io.ErrClosedPipe}, 1)
	assert.ErrorIs(t, err, errWriteFailed)
}

func TestWritePacketRoundTrip(t *testing.T) {
	sizes := []int{1, 2, 254, 255, 256, 511, 960, maxSinglePagePacketSize - 1, maxSinglePagePacketSize}
	samples := []int{120, 240, 480, 960, 5760}

	var buffer bytes.Buffer

	writer, err := NewWith(&buffer, 2)
	require.NoError(t, err)

	written := [][]byte{}
	granule := uint64(0)
	for index, size := range sizes {
		packet := bytes.Repeat([]byte{byte(index + 1)}, size)
		sampleCount := samples[index%len(samples)]
		require.NoError(t, writer.WritePacket(packet, sampleCount))

		written = append(written, packet)
		granule += uint64(sampleCount) //nolint:gosec // sampleCount is positive and tiny.
	}
	require.NoError(t, writer.Close())

	packets, _ := readPackets(t, buffer.Bytes())
	require.Len(t, packets, len(written)+1)
	assert.Equal(t, []byte("OpusTags"), packets[0][:8])
	for index, packet := range written {
		assert.Equal(t, packet, packets[index+1], "packet %d", index)
	}

	pages := parsePages(t, buffer.Bytes())
	assert.Equal(t, byte(pageHeaderTypeBeginningOfStream), pages[0].headerType)
	assert.NotZero(t, pages[len(pages)-1].headerType&pageHeaderTypeEndOfStream)

	// Every audio page must carry the granule of the last packet that
	// completed on it, and the last page must carry the final count.
	assert.Equal(t, granule, pages[len(pages)-1].granule)
}

func TestWritePacketSharesPages(t *testing.T) {
	var buffer bytes.Buffer

	writer, err := NewWith(&buffer, 1)
	require.NoError(t, err)

	const packets = 10
	for range packets {
		require.NoError(t, writer.WritePacket([]byte{0xAA, 0xBB}, 960))
	}
	require.NoError(t, writer.Close())

	pages := parsePages(t, buffer.Bytes())
	// Two header pages plus one audio page holding every packet.
	require.Len(t, pages, 3)
	assert.Len(t, pages[2].lacing, packets)
	assert.Equal(t, uint64(960*packets), pages[2].granule)
	assert.Equal(t, byte(pageHeaderTypeEndOfStream), pages[2].headerType)

	read, _ := readPackets(t, buffer.Bytes())
	require.Len(t, read, packets+1)
	for _, packet := range read[1:] {
		assert.Equal(t, []byte{0xAA, 0xBB}, packet)
	}
}

func TestWritePacketSpansPages(t *testing.T) {
	var buffer bytes.Buffer

	writer, err := NewWith(&buffer, 2)
	require.NoError(t, err)

	// 255 segments of 255 octets fit exactly on one page, so this packet
	// needs 256 segments: the first page holds a continuation marker and the
	// packet completes on the second page.
	packet := make([]byte, maxPagePayload)
	for index := range packet {
		packet[index] = byte(index * 7)
	}
	require.NoError(t, writer.WritePacket(packet, 5760))
	require.NoError(t, writer.WritePacket([]byte{0x11, 0x22}, 960))
	require.NoError(t, writer.Close())

	pages := parsePages(t, buffer.Bytes())
	require.Len(t, pages, 4)
	assert.Equal(t, byte(pageHeaderTypeBeginningOfStream), pages[0].headerType)
	assert.Equal(t, byte(0), pages[1].headerType)

	first := pages[2]
	assert.Equal(t, byte(0), first.headerType) // first page of the packet is not itself continued
	assert.Equal(t, noGranulePosition, first.granule)
	assert.Len(t, first.lacing, maxPageSegments)
	assert.Equal(t, uint32(2), first.sequence)

	second := pages[3]
	assert.Equal(t, byte(pageHeaderTypeContinuedPacket|pageHeaderTypeEndOfStream), second.headerType)
	assert.Len(t, second.lacing, 2) // the long packet's empty terminator, then the short packet
	assert.Equal(t, uint64(5760+960), second.granule)
	assert.Equal(t, []byte{0, 2}, second.lacing)
	assert.Equal(t, []byte{0x11, 0x22}, second.payload)
	assert.Equal(t, uint32(3), second.sequence)

	read, _ := readPackets(t, buffer.Bytes())
	require.Len(t, read, 3)
	assert.Equal(t, packet, read[1])
	assert.Equal(t, []byte{0x11, 0x22}, read[2])
}

func TestWritePacketContinuationFlag(t *testing.T) {
	var buffer bytes.Buffer

	writer, err := NewWith(&buffer, 1)
	require.NoError(t, err)

	// Five full segments short of the page limit.
	head := maxPagePayload - 5*maxPageSegmentSize
	require.NoError(t, writer.WritePacket(make([]byte, head), 960))
	// This packet starts on the same page and spills onto the next one.
	require.NoError(t, writer.WritePacket(make([]byte, 10*maxPageSegmentSize), 1920))
	require.NoError(t, writer.Close())

	pages := parsePages(t, buffer.Bytes())
	require.Len(t, pages, 4)
	assert.Len(t, pages[2].lacing, maxPageSegments)
	// The first packet completed on this page before the second one spilled
	// over, so the page keeps the granule of that completed packet.
	assert.Equal(t, uint64(960), pages[2].granule)
	assert.Equal(t, byte(pageHeaderTypeContinuedPacket|pageHeaderTypeEndOfStream), pages[3].headerType)
	assert.Equal(t, uint64(960+1920), pages[3].granule)

	read, _ := readPackets(t, buffer.Bytes())
	require.Len(t, read, 3)
	assert.Len(t, read[1], head)
	assert.Len(t, read[2], 10*maxPageSegmentSize)
}

func TestLastPacketTrimming(t *testing.T) {
	var buffer bytes.Buffer

	writer, err := NewWith(&buffer, 2)
	require.NoError(t, err)
	require.NoError(t, writer.WritePacket([]byte{0x01}, 960))
	// The encoder's final packet decodes to 960 samples but only 400 are
	// real audio; recording the smaller count trims the end of the stream.
	require.NoError(t, writer.WritePacket([]byte{0x02}, 400))
	require.NoError(t, writer.Close())

	pages := parsePages(t, buffer.Bytes())
	last := pages[len(pages)-1]
	assert.Equal(t, uint64(1360), last.granule)
	assert.Equal(t, byte(pageHeaderTypeEndOfStream), last.headerType)

	read, _ := readPackets(t, buffer.Bytes())
	require.Len(t, read, 3)
	assert.Equal(t, []byte{0x02}, read[2])
}

func TestWritePacketErrors(t *testing.T) {
	testCases := []struct {
		name    string
		packet  []byte
		samples int
		err     error
	}{
		{name: "empty packet", packet: nil, samples: 960, err: errEmptyPacket},
		{name: "zero samples", packet: []byte{0x01}, samples: 0, err: errInvalidSampleCount},
		{name: "negative samples", packet: []byte{0x01}, samples: -960, err: errInvalidSampleCount},
		{name: "too many samples", packet: []byte{0x01}, samples: maxPacketSamples + 1, err: errInvalidSampleCount},
		{name: "packet too large", packet: make([]byte, maxPacketSize+1), samples: 960, err: errPacketTooLarge},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			var buffer bytes.Buffer

			writer, err := NewWith(&buffer, 2)
			require.NoError(t, err)
			before := buffer.Len()

			err = writer.WritePacket(testCase.packet, testCase.samples)
			assert.ErrorIs(t, err, testCase.err)
			assert.Equal(t, before, buffer.Len(), "a rejected packet must not write a page")

			// The writer stays usable after a rejected packet.
			require.NoError(t, writer.WritePacket([]byte{0x02}, 960))
			require.NoError(t, writer.Close())
		})
	}
}

func TestWritePacketMaximumSamples(t *testing.T) {
	var buffer bytes.Buffer

	writer, err := NewWith(&buffer, 2)
	require.NoError(t, err)
	require.NoError(t, writer.WritePacket([]byte{0x01}, maxPacketSamples))
	require.NoError(t, writer.WritePacket(make([]byte, maxPacketSize), maxPacketSamples))
	require.NoError(t, writer.Close())

	pages := parsePages(t, buffer.Bytes())
	assert.Equal(t, uint64(maxPacketSamples)*2, pages[len(pages)-1].granule)
}

func TestWritePacketGranuleOverflow(t *testing.T) {
	var buffer bytes.Buffer

	writer, err := NewWith(&buffer, 2)
	require.NoError(t, err)
	writer.granule = math.MaxUint64 - 1 - 500

	err = writer.WritePacket([]byte{0x01}, 960)
	assert.ErrorIs(t, err, errSampleCountOverflow)

	require.NoError(t, writer.WritePacket([]byte{0x01}, 500))
	assert.Equal(t, uint64(math.MaxUint64-1), writer.granule)
	require.NoError(t, writer.Close())
}

func TestWriteFailure(t *testing.T) {
	// The two header pages take exactly 95 bytes; audio writes fail after that.
	writerOut := &failingWriter{limit: 95, err: io.ErrClosedPipe}
	writer, err := NewWith(writerOut, 1)
	require.NoError(t, err)

	// The short packet stays in the open page until Close, so use a packet
	// that spans a page boundary to force the write here.
	err = writer.WritePacket(make([]byte, maxPagePayload), 960)
	assert.ErrorIs(t, err, errWriteFailed)
	assert.ErrorIs(t, err, io.ErrClosedPipe)

	// The error is sticky: later packets and Close report it rather than
	// silently writing a corrupt stream.
	assert.ErrorIs(t, writer.WritePacket([]byte{0x02}, 960), errWriteFailed)
	assert.ErrorIs(t, writer.Close(), errWriteFailed)
}

func TestShortWrite(t *testing.T) {
	writerOut := &failingWriter{limit: 105}
	writer, err := NewWith(writerOut, 1)
	require.NoError(t, err)

	// The writer accepts only part of the next page; io.ErrShortWrite is
	// surfaced instead of silently dropping the remainder.
	err = writer.WritePacket(make([]byte, maxPagePayload), 960)
	assert.ErrorIs(t, err, errWriteFailed)
	assert.ErrorIs(t, err, io.ErrShortWrite)
}

func TestCloseTwice(t *testing.T) {
	var buffer bytes.Buffer

	writer, err := NewWith(&buffer, 1)
	require.NoError(t, err)
	require.NoError(t, writer.WritePacket([]byte{0x01}, 960))
	require.NoError(t, writer.Close())

	err = writer.Close()
	assert.ErrorIs(t, err, errWriterClosed)

	err = writer.WritePacket([]byte{0x02}, 960)
	assert.ErrorIs(t, err, errWriterClosed)

	// A second Close must not append another page.
	pages := parsePages(t, buffer.Bytes())
	assert.Equal(t, byte(pageHeaderTypeEndOfStream), pages[len(pages)-1].headerType)
	assert.Len(t, pages, 3)
}

func TestNoAudioEndOfStreamPage(t *testing.T) {
	var buffer bytes.Buffer

	writer, err := NewWith(&buffer, 2)
	require.NoError(t, err)
	require.NoError(t, writer.Close())

	pages := parsePages(t, buffer.Bytes())
	require.Len(t, pages, 3)
	assert.Equal(t, byte(pageHeaderTypeEndOfStream), pages[2].headerType)
	assert.Empty(t, pages[2].lacing)
	assert.Equal(t, uint64(0), pages[2].granule)

	packets, _ := readPackets(t, buffer.Bytes())
	assert.Len(t, packets, 1)
}

// TestEncodedPacketsRoundTrip encodes real PCM, writes the packets, reads them
// back through oggreader, and checks that decoding is bit-identical.
func TestEncodedPacketsRoundTrip(t *testing.T) {
	encoder, err := opus.NewEncoder(opus.WithChannels(2))
	require.NoError(t, err)

	frameBytes := make([]byte, 4000)
	pcm := make([]byte, 960*2*2)
	for index := range 960 {
		value := int16(8000 * math.Sin(2*math.Pi*440*float64(index)/48000))
		bits := uint16(value) //nolint:gosec // test signal bit pattern.
		binary.LittleEndian.PutUint16(pcm[index*4:], bits)
		binary.LittleEndian.PutUint16(pcm[index*4+2:], bits)
	}

	packets := [][]byte{}
	for range 3 {
		size, encodeErr := encoder.Encode(pcm, frameBytes)
		require.NoError(t, encodeErr)
		packets = append(packets, append([]byte(nil), frameBytes[:size]...))
	}

	var buffer bytes.Buffer

	writer, err := NewWith(&buffer, 2, WithPreSkip(312))
	require.NoError(t, err)
	for _, packet := range packets {
		require.NoError(t, writer.WritePacket(packet, 960))
	}
	require.NoError(t, writer.Close())

	read, header := readPackets(t, buffer.Bytes())
	assert.Equal(t, uint16(312), header.PreSkip)
	require.Len(t, read, len(packets)+1)
	for index, packet := range packets {
		assert.Equal(t, packet, read[index+1])
	}

	// Decoding the packets stored in the file must reproduce the PCM produced
	// when decoding the originals from a fresh decoder state.
	referenceDecoder := opus.NewDecoder()
	fileDecoder := opus.NewDecoder()
	reference := make([]byte, 1920*2)
	decoded := make([]byte, 1920*2)
	for index, packet := range packets {
		bandwidth, stereo, decodeErr := referenceDecoder.Decode(packet, reference)
		require.NoError(t, decodeErr)
		_, _, decodeErr = fileDecoder.Decode(read[index+1], decoded)
		require.NoError(t, decodeErr)
		assert.Equal(t, opus.BandwidthFullband, bandwidth)
		assert.True(t, stereo)
		assert.Equal(t, reference, decoded, "packet %d", index)
	}
}

func TestPageChecksumsAndSequence(t *testing.T) {
	var buffer bytes.Buffer

	writer, err := NewWith(&buffer, 1, WithSerial(42))
	require.NoError(t, err)
	for index := range 5 {
		require.NoError(t, writer.WritePacket(bytes.Repeat([]byte{byte(index)}, 100), 960))
	}
	require.NoError(t, writer.WritePacket(make([]byte, maxPagePayload), 960))
	require.NoError(t, writer.Close())

	// oggreader.NewWith and ParseNextPacket validate the checksum of every
	// page they read.
	read, _ := readPackets(t, buffer.Bytes())
	require.Len(t, read, 7)

	pages := parsePages(t, buffer.Bytes())
	for index, page := range pages {
		assert.Equal(t, uint32(42), page.serial)
		assert.Equal(t, uint32(index), page.sequence) //nolint:gosec // small loop index.
	}
}

// TestPageGranuleFollowsCompletedPackets checks the RFC 7845 Section 4 rule:
// an audio page's granule position is the sum of the durations of the packets
// that complete on it plus the most recent page with completed packets, and a
// page on which no packet completes carries the all-ones marker.
func TestPageGranuleFollowsCompletedPackets(t *testing.T) {
	var buffer bytes.Buffer

	writer, err := NewWith(&buffer, 2)
	require.NoError(t, err)

	packets := [][]byte{
		bytes.Repeat([]byte{0x01}, 100),
		bytes.Repeat([]byte{0x02}, maxPagePayload), // spans two pages
		bytes.Repeat([]byte{0x03}, 5),
		bytes.Repeat([]byte{0x04}, maxPagePayload),
		bytes.Repeat([]byte{0x05}, 300),
		bytes.Repeat([]byte{0x06}, 1),
	}
	samples := []int{960, 1920, 480, 2880, 960, 120}
	for index, packet := range packets {
		require.NoError(t, writer.WritePacket(packet, samples[index]))
	}
	require.NoError(t, writer.Close())

	pages := parsePages(t, buffer.Bytes())
	audio := pages[2:]
	require.NotEmpty(t, audio)

	granule := uint64(0)
	nextPacket := 0
	for pageIndex, page := range audio {
		completed := false
		for _, lacing := range page.lacing {
			if lacing == maxPageSegmentSize {
				continue
			}
			require.Less(t, nextPacket, len(samples))
			granule += uint64(samples[nextPacket]) //nolint:gosec // sample counts are positive.
			nextPacket++
			completed = true
		}

		if completed {
			assert.Equal(t, granule, page.granule, "page %d", pageIndex)
		} else {
			assert.Equal(t, noGranulePosition, page.granule, "page %d", pageIndex)
		}
	}
	assert.Equal(t, len(samples), nextPacket)
	assert.NotZero(t, audio[len(audio)-1].headerType&pageHeaderTypeEndOfStream)
}

func TestWriterCopiesPacketData(t *testing.T) {
	var buffer bytes.Buffer

	writer, err := NewWith(&buffer, 1)
	require.NoError(t, err)

	packet := []byte{0x01, 0x02, 0x03}
	require.NoError(t, writer.WritePacket(packet, 960))
	packet[0] = 0xFF // the caller may reuse its buffer
	require.NoError(t, writer.Close())

	read, _ := readPackets(t, buffer.Bytes())
	require.Len(t, read, 2)
	assert.Equal(t, []byte{0x01, 0x02, 0x03}, read[1])
}

func TestCloseWriteFailure(t *testing.T) {
	// The two header pages are 95 bytes and the single audio page is 30, so
	// stopping one byte short makes exactly the final page fail.
	writerOut := &failingWriter{limit: 95 + pageHeaderSize, err: io.ErrClosedPipe}
	writer, err := NewWith(writerOut, 1)
	require.NoError(t, err)
	require.NoError(t, writer.WritePacket([]byte{0x01, 0x02}, 960))

	err = writer.Close()
	assert.ErrorIs(t, err, errWriteFailed)
	assert.ErrorIs(t, err, io.ErrClosedPipe)

	// The failure is remembered for a later Close call.
	assert.ErrorIs(t, writer.Close(), errWriterClosed)
}

func TestMaximumSizeVendorSignature(t *testing.T) {
	var buffer bytes.Buffer

	vendor := strings.Repeat("v", maxVendorLength)
	writer, err := NewWith(&buffer, 2, WithVendor(vendor))
	require.NoError(t, err)
	require.NoError(t, writer.Close())

	packets, _ := readPackets(t, buffer.Bytes())
	require.Len(t, packets, 1)
	vendorLength := binary.LittleEndian.Uint32(packets[0][8:12])
	assert.Equal(t, uint32(maxVendorLength), vendorLength)
	assert.Equal(t, vendor, string(packets[0][12:12+vendorLength]))
}

func TestPageSequenceOverflow(t *testing.T) {
	var buffer bytes.Buffer

	writer, err := NewWith(&buffer, 1)
	require.NoError(t, err)
	writer.seq = math.MaxUint32

	err = writer.WritePacket(make([]byte, maxPagePayload), 960)
	assert.ErrorIs(t, err, errPageSequenceOverflow)

	// The failure is sticky rather than silently wrapping to page zero.
	assert.ErrorIs(t, writer.WritePacket([]byte{0x01}, 960), errPageSequenceOverflow)
	assert.ErrorIs(t, writer.Close(), errPageSequenceOverflow)
}

func TestMaximumPacketSizeSpansExactPageCount(t *testing.T) {
	var buffer bytes.Buffer

	writer, err := NewWith(&buffer, 1)
	require.NoError(t, err)
	// One byte over the limit must be rejected up front.
	err = writer.WritePacket(make([]byte, maxPacketSize+1), 960)
	assert.ErrorIs(t, err, errPacketTooLarge)

	require.NoError(t, writer.WritePacket(make([]byte, maxPacketSize), 960))
	require.NoError(t, writer.Close())

	pages := parsePages(t, buffer.Bytes())
	require.Len(t, pages, maxPacketPages+2) // two headers, maxPacketPages data pages
	assert.Equal(t, uint64(960), pages[len(pages)-1].granule)

	read, _ := readPackets(t, buffer.Bytes())
	require.Len(t, read, 2)
	assert.Len(t, read[1], maxPacketSize)
}
