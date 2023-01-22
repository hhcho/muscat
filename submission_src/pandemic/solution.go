package main

import (
	"log"
	"os"
)

func main() {
	// RunGWAS()
	command := os.Args[1]
	out_path := os.Args[2]
	_ = out_path
	role := os.Args[3]

	var err error

	switch command {
	case "keygenInit":
		log.Println("Generation of Local Public Key Share by the Clients")
		err = InitializeCryptoAndPkshareInit(role == "server")
		if err != nil {
			log.Fatalf("Error in keygen: %+v", err)
		}
	case "keygenAggregate":
		log.Println("Get Public key: ", role == "server", " or client")
		err = InitializeCryptoAndPkshareAggregate(role == "server")
		//test := []byte(">>>>> Hello from Go ......................................................................")
		//if err = os.WriteFile(out_path, test, 0644); err != nil {
		//	log.Fatalf("Error in keygen: %+v", err)
		//}
		//log.Println(string(test))
	}
}
