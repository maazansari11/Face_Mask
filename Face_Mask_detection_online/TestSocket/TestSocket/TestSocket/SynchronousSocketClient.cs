using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Net;
using System.Net.Sockets;

namespace TestSocket
{
    
    public class SynchronousSocketClient
    {

        public static void StartClient()
        {
            // Data buffer for incoming data.  
            byte[] bytes = new byte[9048];

            // Connect to a remote device.  

            try
            {
                // Establish the remote endpoint for the socket.  
                // This example uses port 11000 on the local computer.  
                IPHostEntry ipHostInfo = Dns.GetHostEntry(Dns.GetHostName());
                IPAddress ipAddress = ipHostInfo.AddressList[2];
                IPEndPoint remoteEP = new IPEndPoint(ipAddress, 5005);

                // Create a TCP/IP  socket.  
                Socket sender = new Socket(ipAddress.AddressFamily,
                    SocketType.Stream, ProtocolType.Tcp);

                // Connect the socket to the remote endpoint. Catch any errors.  
                try
                {
                    sender.Connect(remoteEP);

                    Console.WriteLine("Socket connected to {0}",
                        sender.RemoteEndPoint.ToString());

                    // Encode the data string into a byte array.  
                    byte[] msg = Encoding.ASCII.GetBytes("0");

                    // Send the data through the socket.  
                    int bytesSent = sender.Send(msg);
                    string dataStr="";
                    // Receive the response from the remote device.  
                    while (true)
                    {
                        int bytesRec = sender.Receive(bytes);
                        //Console.WriteLine("Echoed test = {0}",bytesRec);
                        dataStr += Encoding.ASCII.GetString(bytes, 0, bytesRec);
                        //Console.WriteLine(dataStr);
                        /*if (bytes[bytesRec - 4] == 'E' & bytes[bytesRec - 3] == 'N' & bytes[bytesRec - 2] == 'D' & bytes[bytesRec - 1] == '!')
                        {
                            Console.WriteLine("Echoed test = {0}", Encoding.ASCII.GetString(bytes, 0, bytesRec));
                            Console.WriteLine("end found " + bytesRec);
                            break;
                        }*/
                        if (dataStr.Contains("END!"))
                        {
                            int index2 = dataStr.IndexOf("END!");
                            Console.WriteLine("End found at::::"+index2);
                            int totLen = dataStr.Length;
                            Console.WriteLine("length::::" + dataStr.Length);
                            //string mainData = dataStr.Substring(0, index2);
                            //if (index2 == mainData.Length)
                            //{
                            //    dataStr = dataStr.Substring(0, index2);
                            //}
                            //else
                            //{
                            string mainData = dataStr.Substring(0, index2);
                            Console.WriteLine("main data length::::" + mainData.Length);
                            if (dataStr.Length == index2 + 4)
                            {
                                dataStr = "";
                            }
                            else if (dataStr.Length > (index2+4))
                            {
                                Console.WriteLine("TEST TEST");
                                Console.WriteLine("TEST TEST" + (index2 + 4));
                                string dataStr1 = dataStr.Substring(index2 + 4, (totLen - (index2 + 4)));
                                dataStr = dataStr1;
                            }
                            else
                            {

                            }
                            //}
                            Console.WriteLine("remaining string::" + dataStr);
                        }

                    }

                    // Release the socket.  
                    sender.Shutdown(SocketShutdown.Both);
                    sender.Close();

                }
                catch (ArgumentNullException ane)
                {
                    Console.WriteLine("ArgumentNullException : {0}", ane.ToString());

                }
                catch (SocketException se)
                {
                    Console.WriteLine("SocketException : {0}", se.ToString());
                }
                catch (Exception e)
                {
                    Console.WriteLine("Unexpected exception : {0}", e.ToString());
                }

            }
            catch (Exception e)
            {
                Console.WriteLine(e.ToString());
            }
        }

       
    }
    }

