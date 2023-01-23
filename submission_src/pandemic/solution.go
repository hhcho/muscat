package main

import (
	"encoding/binary"
	"log"
	"os"
	"strconv"

	"github.com/hhcho/petchal/crypto"
	"github.com/hhcho/petchal/mheexample"
	"github.com/hhcho/petchal/mpc"
	"github.com/ldsec/lattigo/v2/ckks"
)

func main() {
	// RunGWAS()
	command := os.Args[1]

	log.Println(command)

	switch command {
	case "setup":

		numClients, _ := strconv.Atoi(os.Args[2])
		smallDim := 100

		cpsList := mpc.NewCryptoParamsForNetwork(ckks.DefaultParams[ckks.PN14QP438], numClients, smallDim, 1)

		for i, cps := range cpsList {
			crypto.SaveCryptoParamsAndRotKeys(i, os.Args[i+3], cps.Sk, cps.AggregateSk, cps.Pk, cps.Rlk, cps.RotKs)
		}

	case "cps-test":

		mheexample.ClientEncryptVectorSimple(os.Args[2])

	case "encrypt-vec-int":

		pid_path := os.Args[2]
		inFile := os.Args[3]
		outFile := pid_path + "/output.bin"

		cps := crypto.NewCryptoParamsFromDiskPath(false, pid_path, 1)

		b, err := os.ReadFile(inFile)
		if err != nil {
			log.Fatal(err)
		}

		log.Println(b)

		numBytes := 8

		vec := make([]float64, len(b)/numBytes)
		for i := range vec {
			vec[i] = float64(int64(binary.LittleEndian.Uint64(b[i*numBytes : (i+1)*numBytes])))
		}

		log.Println(vec)

		vecEnc, _ := crypto.EncryptFloatVector(cps, vec)

		crypto.SaveCipherMatrixToFile(cps, crypto.CipherMatrix{vecEnc}, outFile)

	case "decrypt-test":

		pid_path := os.Args[2]
		inFile := os.Args[3]

		cps := crypto.NewCryptoParamsFromDiskPath(false, pid_path, 1)

		matDec, _ := crypto.LoadCipherMatrixFromFile(cps, inFile)

		out := crypto.DecryptFloatVector(cps, matDec[0], 5)

		log.Println(out)

	case "aggregate-cipher":

		pid_path := os.Args[2]
		inFiles := os.Args[3:]
		outFile := pid_path + "/output.bin"

		cps := crypto.NewCryptoParamsFromDiskPath(true, pid_path, 1)

		var out crypto.CipherMatrix
		for _, f := range inFiles {
			matDec, _ := crypto.LoadCipherMatrixFromFile(cps, f)
			if out == nil {
				out = matDec
			} else {
				for i := range matDec {
					out[i] = crypto.CAdd(cps, out[i], matDec[i])
				}
			}
		}

		crypto.SaveCipherMatrixToFile(cps, out, outFile)

	case "decrypt-client-send":

		pidPath := os.Args[2]
		inFile := os.Args[3]

		mheexample.CollectiveDecryptClientSend(pidPath, inFile)

	case "decrypt-server":

		pidPath := os.Args[2]
		nr, _ := strconv.Atoi(os.Args[3])
		nc, _ := strconv.Atoi(os.Args[4])
		inFiles := os.Args[5:]

		mheexample.CollectiveDecryptServer(pidPath, nr, nc, inFiles)

	case "decrypt-client-receive":

		pidPath := os.Args[2]
		inFile := os.Args[3]
		serverFile := os.Args[4]

		mheexample.CollectiveDecryptClientReceive(pidPath, inFile, serverFile)

	}
}
