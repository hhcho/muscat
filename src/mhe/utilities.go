package mhe

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"unsafe"

	"github.com/ldsec/lattigo/v2/ckks"
	"log"
)

// Max computes the maximum values between two integers
func Max(x, y int) int {
	if x <= y {
		return y
	}
	return x
}

// CAdd additions two Ciphervectors
func CAdd(cryptoParams *CryptoParams, X CipherVector, Y CipherVector) CipherVector {
	res := make(CipherVector, len(X)) //equal num of ciphertexts
	cryptoParams.WithEvaluator(func(eval ckks.Evaluator) error {
		for i := 0; i < Max(len(Y), len(X)); i++ {
			//check level
			res[i] = eval.AddNew(X[i], Y[i])
		}
		return nil
	})
	return res
}

// MarshalCipherMatrix returns byte array corresponding to ciphertext sizes (int array) and byte array corresponding to marshaling
func MarshalCipherMatrix(cm CipherMatrix) ([]byte, []byte) {
	cmBytes, ctSizes, err := cm.MarshalBinary()
	if err != nil {
		panic(err)
	}

	r, c := len(cm), len(cm[0])
	intsize := uint64(unsafe.Sizeof(ctSizes[0][0]))
	sizesbuf := make([]byte, intsize*uint64(r)*uint64(c))

	offset := uint64(0)
	for i := range ctSizes {
		for j := range ctSizes[i] {
			binary.LittleEndian.PutUint64(sizesbuf[offset:offset+intsize], uint64(ctSizes[i][j]))
			offset += intsize
		}

	}

	return sizesbuf, cmBytes

}

// UnmarshalCipherMatrix returns byte array corresponding to ciphertext sizes (int array) and byte array corresponding to marshaling
func UnmarshalCipherMatrix(cryptoParams *CryptoParams, r, c int, sbytes, ctbytes []byte) CipherMatrix {
	intsize := uint64(8)
	offset := uint64(0)
	sizes := make([][]int, r)
	for i := range sizes {
		sizes[i] = make([]int, c)
		for j := range sizes[i] {
			sizes[i][j] = int(binary.LittleEndian.Uint64(sbytes[offset:]))
			offset += intsize

		}
	}

	cm := make(CipherMatrix, 1)
	err := (&cm).UnmarshalBinary(cryptoParams, ctbytes, sizes)
	if err != nil {
		panic(err)

	}

	return cm
}

// SaveCipherMatrixToFile saves a Ciphermatrix to a file
func SaveCipherMatrixToFile(cps *CryptoParams, cm CipherMatrix, filename string) {
	file, err := os.Create(filename)
	defer file.Close()
	if err != nil {
		log.Fatal(err)
	}

	writer := bufio.NewWriter(file)

	sbytes, cmbytes := MarshalCipherMatrix(cm)

	nrbuf := make([]byte, 4)
	ncbuf := make([]byte, 4)
	binary.LittleEndian.PutUint32(nrbuf, uint32(len(cm)))
	binary.LittleEndian.PutUint32(ncbuf, uint32(len(cm[0])))

	sbuf := make([]byte, 8)
	cmbuf := make([]byte, 8)
	binary.LittleEndian.PutUint64(sbuf, uint64(len(sbytes)))
	binary.LittleEndian.PutUint64(cmbuf, uint64(len(cmbytes)))

	writer.Write(nrbuf)
	writer.Write(ncbuf)
	writer.Write(sbuf)
	writer.Write(sbytes)
	writer.Write(cmbuf)
	writer.Write(cmbytes)

	writer.Flush()
}

// LoadCipherMatrixFromFile reads a ciphermatrix from a file
func LoadCipherMatrixFromFile(cps *CryptoParams, filename string) (CipherMatrix, error) {
	file, err := os.Open(filename)
	defer file.Close()
	if err != nil {
		log.Println(err)
	}

	reader := bufio.NewReader(file)

	ibuf := make([]byte, 4)
	io.ReadFull(reader, ibuf)
	nrows := int(binary.LittleEndian.Uint32(ibuf))
	io.ReadFull(reader, ibuf)
	numCtxPerRow := int(binary.LittleEndian.Uint32(ibuf))

	sbuf := make([]byte, 8)
	io.ReadFull(reader, sbuf)
	sbyteSize := binary.LittleEndian.Uint64(sbuf)
	sdata := make([]byte, sbyteSize)
	io.ReadFull(reader, sdata)

	cmbuf := make([]byte, 8)
	io.ReadFull(reader, cmbuf)
	cbyteSize := binary.LittleEndian.Uint64(cmbuf)
	cdata := make([]byte, cbyteSize)
	io.ReadFull(reader, cdata)

	return UnmarshalCipherMatrix(cps, nrows, numCtxPerRow, sdata, cdata), err
}

// SaveFloatVectorToFileBinary saves a vector of float values to a binary file
func SaveFloatVectorToFileBinary(filename string, x []float64) {
	file, err := os.Create(filename)
	defer file.Close()
	if err != nil {
		log.Fatal(err)
	}

	writer := bufio.NewWriter(file)

	for i := range x {
		binary.Write(writer, binary.LittleEndian, x[i])
	}

	writer.Flush()
}

// SaveFloatVectorToFile saves a vector of float values to a file
func SaveFloatVectorToFile(filename string, x []float64) {
	file, err := os.Create(filename)
	defer file.Close()
	if err != nil {
		log.Fatal(err)
	}

	writer := bufio.NewWriter(file)

	for i := range x {
		writer.WriteString(fmt.Sprintf("%.6e\n", x[i]))
	}

	writer.Flush()
}

// LoadFloatVectorFromFile reads a vector of float values from a file
func LoadFloatVectorFromFile(filename string, n int) []float64 {
	file, err := os.Open(filename)
	defer file.Close()
	if err != nil {
		log.Fatal(err)
	}

	reader := bufio.NewReader(file)

	out := make([]float64, n)

	for i := range out {
		line, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}

		_, err = fmt.Sscanf(line, "%f", &out[i])
		if err != nil {
			log.Fatal(err)
		}
	}

	return out
}
