package main

import (
	"bufio"
	"encoding/binary"
	"github.com/aead/chacha20/chacha"
	"github.com/dinvlad/pets-private/mpc"
	"github.com/hhcho/frand"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/drlwe"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/utils"
	"go.dedis.ch/onet/v3/log"
	"io"
	"os"
	"strconv"
)

var ringQPFileName = "ringQP.bin"
var pkShareFileName = "pkshare"
var pkFileName = "pk.bin"

func InitializeCryptoAndPkshareInit(isServer bool) error {

	// TODO ??
	sharedKeysPath := ""

	// setup default CKKS params
	params := ckks.DefaultParams[ckks.PN14QP438]

	dckksContext := dckks.NewContext(params)

	// REF: CollectiveInit
	var kgen = ckks.NewKeyGenerator(params)

	var skShard *ckks.SecretKey

	// save RingQP state
	RingQPBytes, err := dckksContext.RingQP.MarshalBinary()
	if err != nil {
		log.Error("Couldn't save ringQP state")
	}
	WriteFullFile(ringQPFileName, RingQPBytes)

	if isServer { // Not sure that this is needed
		skShard = new(ckks.SecretKey)
		skShard.Value = dckksContext.RingQP.NewPoly()
	} else { // client
		skShard = kgen.GenSecretKey()

		skShard.Value.Zero()
		prng, err := utils.NewPRNG() // Use NewKeyedPRNG for debugging if deterministic behavior is desired
		if err != nil {
			panic(err)
		}
		ternarySamplerMontgomery := ring.NewTernarySampler(prng, dckksContext.RingQP, 1.0/3.0, true)
		skShard.Value = ternarySamplerMontgomery.ReadNew()
		dckksContext.RingQP.NTT(skShard.Value, skShard.Value)

	}

	// Globally shared random generator, see whether we need real inputs
	netObj := mpc.Network{Rand: mpc.InitializePRG(1, 1, sharedKeysPath)}

	seed := make([]byte, chacha.KeySize)
	netObj.Rand.SwitchPRG(-1)
	netObj.Rand.RandRead(seed)
	netObj.Rand.RestorePRG()

	seedPrng := frand.NewCustom(seed, mpc.BufferSize, 20)

	crpGen := ring.NewUniformSamplerWithBasePrng(seedPrng, dckksContext.RingQP)

	// REF CollectivePubKeyGen
	sk := &skShard.SecretKey

	ckgProtocol := dckks.NewCKGProtocol(params)

	pkShare := ckgProtocol.AllocateShares()

	crp := crpGen.ReadNew()
	ckgProtocol.GenShare(sk, crp, pkShare)

	// need aggregation for Public Key
	// REF AggregatePubKeyShares
	//out := new(drlwe.CKGShare)

	if !isServer {
		bytes, err := pkShare.MarshalBinary()
		if err != nil {
			log.Error("Could not marshall pkshare: ", err)
		}
		//buf := make([]byte, 4)
		//binary.LittleEndian.PutUint32(buf, uint32(len(bytes)))
		return WriteFullFile(pkShareFileName, bytes)
	}

	return nil
}

func InitializeCryptoAndPkshareAggregate(isServer bool) error {
	// REF AggregatePubKeyShares
	ringQP := new(ring.Ring)
	if isServer {
		out := new(drlwe.CKGShare)
		// get back RingQP
		ringQPBytes, _ := LoadFullFile(ringQPFileName)
		ringQP.UnmarshalBinary(ringQPBytes)

		// read and aggregate
		fileIndex := 0
		receivedPkShare := new(ring.Poly)
		for pkshareByte, err := LoadFullFile(pkShareFileName + strconv.Itoa(fileIndex)); err == nil; {
			receivedPkShare.UnmarshalBinary(pkshareByte)
			ringQP.Add(receivedPkShare, out.Poly, out.Poly)
			fileIndex++
		}
		pkbytes, err := out.Poly.MarshalBinary()
		if err != nil {
			log.Error("could not marshall aggregated public key")
		}
		err = WriteFullFile(pkFileName, pkbytes)
		return err
	}
	return nil
}

func WriteFullFile(filename string, buf []byte) error {
	file, err := os.Create(filename)
	defer file.Close()
	if err != nil {
		log.Fatal(err)
	}
	sbuf := make([]byte, 8)
	binary.LittleEndian.PutUint64(sbuf, uint64(len(buf)))
	writer := bufio.NewWriter(file)
	writer.Write(buf)
	writer.Flush()

	return err
}

func LoadFullFile(filename string) ([]byte, error) {
	file, err := os.Open(filename)
	defer file.Close()
	if err != nil {
		return nil, err
	}

	reader := bufio.NewReader(file)

	sbuf := make([]byte, 8)
	io.ReadFull(reader, sbuf)
	sbyteSize := binary.LittleEndian.Uint64(sbuf)
	sdata := make([]byte, sbyteSize)
	io.ReadFull(reader, sdata)

	return sdata, err
}
